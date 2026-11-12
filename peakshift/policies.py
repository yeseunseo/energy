"""운영 방식 네 가지.

(A) EDDPolicy      기존 운영: 납기 순, 전력 고려 없음, 근무 중엔 설비를 켜둠
(B) PlanPolicy     하루 전 CP-SAT 계획을 따르고, 계획에 없는 상황은 규칙으로 수리 (= 롤아웃의 기본 정책)
(C) MPCPolicy      이벤트마다 남은 시간에 대해 CP-SAT 재계획
(D) RolloutPolicy  B를 기본 정책으로 한 시뮬레이션 기반 롤아웃 (설비별 순차 결정)
"""
from __future__ import annotations

import numpy as np

from . import instance as I
from .instance import Instance
from . import sim as S
from . import planner


# ---------------------------------------------------------------- (A)

def _work_left(st, m) -> bool:
    """이 기계가 앞으로 할 공정이 남아 있는지."""
    return any(st.op_status[o] != S.DONE and m in st.ops[o].proc and st.op_m[o] in (-1, m) for o in range(len(st.ops)))


class EDDPolicy:
    name = 'A 기존 운영(EDD)'

    def decide(self, st: S.State, inst: Instance) -> dict:
        acts, taken = {}, set()
        for m in S.free_machines(st):
            cands = [o for o in S.ready_ops(st, inst, m) if o not in taken]
            if cands:
                o = min(cands, key=lambda o: (st.jobs[st.ops[o].job].due, o))
                acts[m] = ('start', o)
                taken.add(o)
            elif st.m_state[m] == S.IDLE and (not inst.labor.regular[st.t] or not _work_left(st, m)):
                acts[m] = ('off',)
        return acts


# ---------------------------------------------------------------- (B) 기본 정책

class PlanPolicy:
    """계획 추종 + 규칙 수리.

    규칙: ① 계획에 없는 공정(급주문 등)은 남은 일이 가장 적은 적격 설비 큐에 납기 여유 순으로 넣는다.
          ② 고장으로 오래 멈춘 설비의 큐는 다른 적격 설비로 옮긴다.
          ③ 계획 시각이 되면 투입하되, 투입하면 부하가 기준치 P̄를 넘고 납기 여유가 있으면 미룬다.
          ④ 다음 투입까지 간격이 예열보다 충분히 길면 설비를 끈다.
    """
    name = 'B 정적 최적화'

    def __init__(self, plan: planner.Plan | None = None, slack_guard: int | None = None, pv_hint=None,
                 bat_mode: str = 'track'):
        self.plan = plan
        self.bat_mode = bat_mode   # 'track': 계획한 한전 전력에 맞추도록 배터리가 차이를 메움 / 'follow': 계획한 배터리 출력 그대로
        self.slack_guard = slack_guard if slack_guard is not None else I.slot_of(1.0)   # 1시간
        self.pv_hint = pv_hint  # 현재 슬롯 발전량 추정용 (없으면 상태의 pv 사용)

    def install(self, st: S.State):
        st.queues = [list(q) for q in self.plan.queues]
        st.pstart = dict(self.plan.pstart)
        st.u_plan = self.plan.u
        st.g_target, st.b_plan, st.bat_mode = self.plan.g_plan, self.plan.b_plan, self.bat_mode
        for o in self.plan.skipped:
            if o not in st.unassigned:
                st.unassigned.append(o)

    def _queued_work(self, st, m):
        return sum(S.proc_time(st, o, m) for o in st.queues[m] if st.op_status[o] == S.WAIT)

    def _repair(self, st: S.State, inst: Instance):
        # ① 미배정 공정 배정
        for o in list(st.unassigned):
            if st.op_status[o] != S.WAIT:
                st.unassigned.remove(o)
                continue
            ms = [m for m in st.ops[o].proc if S.eligible(st, o, m)]
            m = min(ms, key=lambda m: (st.m_state[m] == S.DOWN, self._queued_work(st, m)))
            q = st.queues[m]
            sl = S.slack(st, o)
            pos = next((i for i, oo in enumerate(q) if st.op_status[oo] == S.WAIT and S.slack(st, oo) > sl), len(q))
            q.insert(pos, o)
            st.pstart[o] = st.t
            st.unassigned.remove(o)
        # ② 고장 설비 큐 이관
        for m in range(len(inst.machines)):
            if st.m_state[m] == S.DOWN and st.m_down_until[m] - st.t > 4 and st.queues[m]:
                for o in list(st.queues[m]):
                    alts = [mm for mm in st.ops[o].proc if mm != m and st.m_state[mm] != S.DOWN]
                    if st.op_status[o] == S.WAIT and st.op_m[o] == -1 and alts:
                        mm = min(alts, key=lambda mm: self._queued_work(st, mm))
                        st.queues[m].remove(o)
                        st.queues[mm].append(o)
                        st.pstart[o] = max(st.pstart.get(o, st.t), st.t)

    def action_for(self, st: S.State, inst: Instance, m: int, taken: set, load_now: float) -> tuple:
        lead = inst.machines[m].warmup if st.m_state[m] == S.OFF else 0
        q = [o for o in st.queues[m] if st.op_status[o] == S.WAIT]
        for o in q[:2]:
            if o in taken or not S.pred_done(st, o) or not S.can_start(st, inst, o, m):
                continue
            if st.t + lead < st.pstart.get(o, st.t):
                break
            # ③ 피크 규칙
            add = st.ops[o].proc[m][1] - (inst.machines[m].p_idle if st.m_state[m] == S.IDLE else 0)
            pv = st.pv[st.t] if self.pv_hint is None else self.pv_hint[st.t]
            if load_now + add - pv > inst.P_ref and S.slack(st, o) > self.slack_guard:
                break
            return ('start', o)
        # 예열 선행: 계획상 켜져 있어야 하는 시간이면, 앞 공정이 끝나기 전이라도 미리 켜 둔다
        if st.m_state[m] == S.OFF and q and st.u_plan is not None and st.u_plan[m][st.t] == 1 \
                and inst.labor.window_ok(st.t + inst.machines[m].warmup, 1):
            return ('on',)
        # ④ 끄기
        if st.m_state[m] == S.IDLE:
            nxt = min((st.pstart.get(o, st.t) for o in q), default=I.SLOTS)
            gap = nxt - st.t
            plan_off = st.u_plan is not None and st.u_plan[m][st.t] == 0
            if gap > inst.machines[m].warmup + 2 and (plan_off or not q):
                return ('off',)
        return ('wait',)

    def decide(self, st: S.State, inst: Instance) -> dict:
        self._repair(st, inst)
        acts, taken = {}, set()
        load_now = S.current_load(st, inst)
        for m in S.free_machines(st):
            a = self.action_for(st, inst, m, taken, load_now)
            acts[m] = a
            if a[0] == 'start':
                o = a[1]
                taken.add(o)
                load_now += st.ops[o].proc[m][1]
        return acts


# ---------------------------------------------------------------- (C)

class MPCPolicy:
    name = 'C 재최적화(MPC)'

    def __init__(self, plan, pv_q, time_limit=5.0, workers=1, pv_gap=None):
        self.base = PlanPolicy(plan)
        self.q10, self.q50, self.q90 = pv_q
        self.time_limit, self.workers = time_limit, workers
        self.last_replan = -99
        self.pv_gap = pv_gap if pv_gap is not None else I.slot_of(2.0)   # 2시간
        self.replans = 0

    def install(self, st):
        self.base.install(st)

    def _pv_deviated(self, st):
        t = st.t - 1
        return t >= 1 and all(not (self.q10[k] <= st.pv[k] <= self.q90[k]) for k in (t - 1, t))

    def decide(self, st, inst):
        trigger = st.last_event == st.t or (self._pv_deviated(st) and st.t - self.last_replan >= self.pv_gap)
        if trigger and st.t < I.SLOTS - 2:
            pv_fc = self.q50.copy()
            pv_fc[:st.t] = st.pv[:st.t]
            plan = planner.solve(inst, pv_fc, st=st, time_limit=self.time_limit, workers=self.workers)
            if plan.status in ('OPTIMAL', 'FEASIBLE'):
                st.queues = [list(q) for q in plan.queues]
                st.pstart.update(plan.pstart)
                st.u_plan = plan.u
                if plan.g_plan is not None:
                    st.g_target, st.b_plan = plan.g_plan, plan.b_plan
                st.unassigned = [o for o in plan.skipped if st.op_status[o] == S.WAIT]
                self.replans += 1
            self.last_replan = st.t
        return self.base.decide(st, inst)


# ---------------------------------------------------------------- (D)

class RolloutPolicy:
    """설비별 순차(multiagent) 롤아웃.

    결정 시점마다 빈 설비 하나씩, 후보 행동 {기본 정책 행동, 준비된 공정 상위 k개 투입, 대기, 끄기}를
    같은 N개 시나리오(공통 난수)에서 남은 하루 끝까지 기본 정책으로 시뮬레이션해 평균 총비용이 가장 낮은 것을 고른다.
    앞에서 정한 설비의 행동은 고정하고 다음 설비를 정한다.
    """
    name = 'D PeakShift(롤아웃)'

    def __init__(self, plan, forecaster, cloud_fc, urgent_maker, n_scen=6, k=3, seed=0,
                 urgent_rate=0.15, pv_pool=60, p_spread=0.0, urgent_hours=(9, 17),
                 slack_guard=False, guard_buffer_h=1.0, terminal=None, late_mult=1.0,
                 value_model=None, horizon_h=2.0):
        """slack_guard: 남은 일에 비해 시간이 빠듯한 기계는 '대기·끄기' 후보를 뺌 (D1)
        terminal='realistic': 못 끝낸 작업 = 남은 공정을 그날 평균 가격으로 처리하는 비용 + 지연 비용(작업 평균 비용 × late_mult) (D2)"""
        self.base = PlanPolicy(plan)
        self.slack_guard, self.guard_buffer_h = slack_guard, guard_buffer_h
        self.terminal, self.late_mult = terminal, late_mult
        self.value_model, self.horizon = value_model, I.slot_of(horizon_h)   # 형세 판단: horizon만 굴리고 나머지는 학습 모델로 예측
        self.n, self.k = n_scen, k
        self.rng = np.random.default_rng(seed + 7)
        self.pool = forecaster.scenarios(cloud_fc, pv_pool, self.rng)
        self.q50 = forecaster.predict(cloud_fc)[1]
        self.urgent_maker = urgent_maker   # (rng, slot) -> payload callable
        self.urgent_rate = urgent_rate     # 시간당 급주문 도착 확률 (사전 분포)
        self.p_spread = p_spread           # 처리 시간 변동 폭 (상상하는 미래에도 같은 분포로 뽑음)
        self.urgent_hours = urgent_hours
        self.decisions = 0
        self.sims = 0
        self._job_cost = None
        self.explain = False   # True면 결정마다 후보별 예상치를 기록 (설명용)
        self.log = []

    def install(self, st):
        self.base.install(st)

    def _scenarios(self, st):
        """현재까지의 예보 편차를 반영한 미래 시나리오 N개 (태양광 경로 + 급주문)."""
        t = st.t
        lo = max(0, t - 4)
        bias = float(np.mean(st.pv[lo:t] - self.q50[lo:t])) if t > lo else 0.0
        decay = np.exp(-np.arange(I.SLOTS - t) * I.SLOT_H / 2.0)   # 약 2시간 단위로 줄어듦
        out = []
        for i in self.rng.integers(0, len(self.pool), self.n):
            path = np.array(st.pv, dtype=float)
            path[t:] = np.clip(self.pool[i][t:] + bias * decay, 0, None)
            events = []
            for h in range(max(int(t * I.SLOT_H) + 1, self.urgent_hours[0]), self.urgent_hours[1]):
                if self.urgent_maker is not None and self.rng.random() < self.urgent_rate:
                    slot = I.slot_of(h)
                    events.append((slot, 'urgent', self.urgent_maker(self.rng, slot)))
            pact = None
            if self.p_spread > 0:   # 아직 안 끝난 공정의 실제 처리 시간을 상상 (진행 중이면 이미 지난 시간보다 길게)
                pact = {}
                for o in range(len(st.ops)):
                    if st.op_status[o] == S.DONE:
                        continue
                    p = list(st.ops[o].proc.values())[0][0]
                    a = max(1, int(round(p * self.rng.uniform(1 - self.p_spread, 1 + self.p_spread))))
                    if st.op_status[o] == S.RUN:
                        a = max(a, st.op_elapsed[o] + 1)
                    pact[o] = a
            out.append((path, events, int(self.rng.integers(1 << 30)), pact))   # 고장 seed · 처리 시간 (후보끼리 공유)
        return out

    def _tight(self, st, m) -> bool:
        """이 기계의 남은 일, 또는 이 기계에서 기다리는 작업의 남은 공정이 (처리 시간이 최대로 늘 경우까지) 남은 시간에 빠듯한지."""
        left = I.SLOTS - st.t - I.slot_of(self.guard_buffer_h)
        grow = 1 + self.p_spread
        work_m = 0
        for o in range(len(st.ops)):
            if st.op_status[o] == S.DONE or m not in st.ops[o].proc or st.op_m[o] not in (-1, m):
                continue
            work_m += st.op_rem[o] if st.op_status[o] == S.RUN else st.ops[o].proc[m][0]
        if work_m * grow >= left:
            return True
        for j in st.jobs:
            rem = [o for o in j.ops if st.op_status[o] != S.DONE]
            if rem and any(m in st.ops[o].proc for o in rem):
                chain = sum(st.op_rem[o] if st.op_status[o] == S.RUN else list(st.ops[o].proc.values())[0][0] for o in rem)
                if chain * grow >= left:
                    return True
        return False

    def _candidates(self, st, inst, m, base_act, taken):
        cands = [base_act]
        ready = [o for o in S.ready_ops(st, inst, m) if o not in taken]
        ready.sort(key=lambda o: S.slack(st, o))
        cands += [('start', o) for o in ready[:self.k]]
        tight = self.slack_guard and ready and self._tight(st, m)
        if not tight:
            cands.append(('wait',))
            if st.m_state[m] == S.IDLE:
                cands.append(('off',))
        elif base_act[0] in ('wait', 'off'):
            cands = cands[1:] or cands      # 빠듯하면 기본 정책의 '대기'도 후보에서 뺌
        uniq = []
        for c in cands:
            if c not in uniq:
                uniq.append(c)
        # 할 일도 없고 켜져 있지도 않으면 고민할 필요 없음
        if st.m_state[m] == S.OFF and not ready:
            return [base_act]
        return uniq

    def _evaluate(self, st, inst, acts, scen, stats=None):
        total = 0.0
        unf = pvu = grid = 0.0
        for path, events, bseed, pact in scen:
            s2 = st.clone(pv=path, events=events, bd_seed=bseed, p_act=pact)
            S.step(s2, inst, acts)
            if self.value_model is not None:
                S.run(s2, inst, self.base, until=min(I.SLOTS, st.t + 1 + self.horizon))
                if s2.t < I.SLOTS:
                    from .value import cost_so_far
                    S.apply_events(s2, inst)
                    total += cost_so_far(s2, inst) + self.value_model.predict(s2, inst, self.q50)
                else:
                    total += self._final_cost(s2, inst)
            else:
                S.run(s2, inst, self.base)
                total += self._final_cost(s2, inst)
            if stats is not None:
                unf += any(s2.op_status[j.ops[-1]] != S.DONE for j in s2.jobs)
                pvu += s2.pv_used - st.pv_used
                grid += s2.grid_kwh - st.grid_kwh
            self.sims += 1
        if stats is not None:
            n = len(scen)
            stats.update(cost=total / n, p_unfinished=unf / n, pv_kwh=pvu / n, grid_kwh=grid / n)
        return total / len(scen)

    def _final_cost(self, s2, inst):
        if self.terminal != 'realistic':
            return S.total_cost(s2, inst)
        c = S.cost_breakdown(s2, inst)
        base = c['total'] - c['unfinished_pen']
        price = float(np.mean(inst.tariff.energy))
        rem_kwh, late = 0.0, 0
        for j in s2.jobs:
            rem = [o for o in j.ops if s2.op_status[o] != S.DONE]
            if rem:
                late += 1
                for o in rem:
                    p, kw = list(s2.ops[o].proc.values())[0]
                    left = s2.op_rem[o] if s2.op_status[o] == S.RUN else p
                    rem_kwh += kw * max(left, 0) * I.SLOT_H
        if self._job_cost is None:
            per = [sum(kw * p for (p, kw) in (list(s2.ops[o].proc.values())[0] for o in j.ops)) for j in s2.jobs]
            self._job_cost = price * float(np.mean(per)) * I.SLOT_H
        return base + price * rem_kwh + self.late_mult * self._job_cost * late

    def decide(self, st, inst):
        base_acts = self.base.decide(st, inst)  # 규칙 수리(큐 갱신)를 실제 상태에 반영
        free = list(base_acts.keys())
        if not free:
            return base_acts
        scen = None
        decided = {}
        taken = set()
        for m in free:
            cands = self._candidates(st, inst, m, base_acts[m], taken)
            if len(cands) == 1:
                decided[m] = cands[0]
            else:
                if scen is None:
                    scen = self._scenarios(st)
                    self.decisions += 1
                best, best_c = None, float('inf')
                rec = [] if self.explain else None
                for a in cands:
                    acts = {}
                    for mm in free:
                        if mm in decided:
                            acts[mm] = decided[mm]
                        elif mm == m:
                            acts[mm] = a
                        else:
                            acts[mm] = base_acts[mm]
                    # 같은 공정을 두 설비가 잡으면 나중 설비는 대기
                    seen = set()
                    for mm in list(acts):
                        if acts[mm][0] == 'start':
                            if acts[mm][1] in seen:
                                acts[mm] = ('wait',)
                            else:
                                seen.add(acts[mm][1])
                    stats = {} if rec is not None else None
                    c = self._evaluate(st, inst, acts, scen, stats)
                    if rec is not None:
                        rec.append((a, stats))
                    if c < best_c - 1e-6:
                        best, best_c = a, c
                decided[m] = best
                if rec is not None:
                    self.log.append(dict(t=st.t, machine=m, base=base_acts[m], chosen=best, options=rec))
            if decided[m][0] == 'start':
                taken.add(decided[m][1])
        # 최종 충돌 정리
        seen = set()
        for m in list(decided):
            if decided[m][0] == 'start':
                if decided[m][1] in seen:
                    decided[m] = ('wait',)
                else:
                    seen.add(decided[m][1])
        return decided


def _label(act, st):
    if act[0] == 'start':
        op = st.ops[act[1]]
        return f'작업 J{op.job + 1}의 {op.seq + 1}번째 공정을 지금 시작'
    return {'wait': '15분 기다림', 'off': '기계를 끔', 'on': '기계를 켬'}.get(act[0], act[0])


def explain_decisions(policy, st, inst, top: int = 5, unit: str = '', scale: float = 1.0) -> list:
    """롤아웃 기록에서 '판단이 가장 크게 갈린' 결정들을 사람이 읽는 문장으로.
    차이가 큰 순서: 선택한 행동과 기본 계획 행동의 예상 비용 차이."""
    out = []
    for rec in policy.log:
        opts = {tuple(a): s for a, s in rec['options']}
        ch, bs = opts.get(tuple(rec['chosen'])), opts.get(tuple(rec['base']))
        if ch is None or bs is None or rec['chosen'] == rec['base']:
            continue
        out.append(((bs['cost'] - ch['cost']) * scale, rec, ch, bs))
    out.sort(key=lambda x: -x[0])
    lines = []
    for gain, rec, ch, bs in out[:top]:
        hh, mm = divmod(rec['t'] * I.SLOT_MIN, 60)
        name = inst.machines[rec['machine']].name
        lines.append(
            f"[{hh:02d}:{mm:02d} · {name}] 계획대로라면 '{_label(rec['base'], st)}' → '{_label(rec['chosen'], st)}' 선택. "
            f"예상 비용 {gain:,.0f}{unit} 절감, 못 끝낼 확률 {bs['p_unfinished']:.0%} → {ch['p_unfinished']:.0%}, "
            f"태양광 사용 {ch['pv_kwh'] - bs['pv_kwh']:+.1f}kWh, 전력망 구매 {ch['grid_kwh'] - bs['grid_kwh']:+.1f}kWh")
    return lines
