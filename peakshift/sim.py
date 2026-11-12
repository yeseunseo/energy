"""15분 슬롯 단위 공장 시뮬레이터.

롤아웃이 한 결정마다 수십 번 복제·실행하므로, 상태는 리스트 기반으로 가볍게 유지한다.
"""
from __future__ import annotations

import numpy as np

from . import instance as I
from .instance import Instance

# 공정 상태
WAIT, RESERVED, RUN, DONE = 0, 1, 2, 3
# 설비 상태
OFF, WARMING, IDLE, BUSY, DOWN = 0, 1, 2, 3, 4


class State:
    __slots__ = ('t', 'ops', 'jobs', 'op_status', 'op_m', 'op_rem', 'op_end',
                 'm_state', 'm_warm', 'm_op', 'm_down_until', 'm_idle_run', 'm_startup_now',
                 'queues', 'pstart', 'u_plan', 'unassigned',
                 'energy', 'P_day', 'labor_used', 'startups', 'pv_used', 'grid_kwh',
                 'events', 'ev_idx', 'last_event', 'pv', 'g_hist', 'load_hist', 'next_jid',
                 'bd', 'bd_rng', 'ton', 'bd_count', 'bd_slots',
                 'soc', 'g_target', 'b_plan', 'bat_mode', 'bat_kwh', 'p_act', 'op_elapsed')

    @classmethod
    def initial(cls, inst: Instance, pv: np.ndarray, events: list, keep_history=True,
                breakdown=None, bd_seed: int = 0, p_act: dict | None = None) -> 'State':
        """breakdown=(θ_on, θ_bd): 오래 돌수록 고장 확률이 커지는 고장 모델 (Shao 외 2023). None이면 정해진 고장 사건만 씀."""
        s = cls()
        s.bd = breakdown
        s.bd_rng = np.random.default_rng(bd_seed)
        s.ton = [0] * len(inst.machines)
        s.bd_count = 0
        s.bd_slots = 0
        s.soc = inst.battery['soc0'] if inst.battery else 0.0
        s.g_target = None     # 계획한 한전 전력 (배터리 'track' 방식에서 이 값에 맞추도록 배터리가 차이를 메움)
        s.b_plan = None       # 계획한 배터리 출력 (+방전 / −충전)
        s.bat_mode = None     # 'track' | 'follow' | 'greedy' (없으면 greedy)
        s.bat_kwh = 0.0
        s.p_act = dict(p_act) if p_act else {}   # 실제 처리 칸 수 (공정이 끝나야 드러남, 정책은 보지 않음)
        s.op_elapsed = [0] * len(inst.ops)
        s.t = 0
        s.ops = list(inst.ops)
        s.jobs = list(inst.jobs)
        n, M = len(s.ops), len(inst.machines)
        s.op_status = [WAIT] * n
        s.op_m = [-1] * n
        s.op_rem = [-1] * n
        s.op_end = [-1] * n
        s.m_state = [OFF] * M
        s.m_warm = [0] * M
        s.m_op = [-1] * M
        s.m_down_until = [0] * M
        s.m_idle_run = [0] * M
        s.m_startup_now = [False] * M
        s.queues = [[] for _ in range(M)]
        s.pstart = {}
        s.u_plan = None
        s.unassigned = []
        s.energy = 0.0
        s.P_day = 0.0
        s.labor_used = set()
        s.startups = 0
        s.pv_used = 0.0
        s.grid_kwh = 0.0
        s.events = sorted(events, key=lambda e: e[0])
        s.ev_idx = 0
        s.last_event = -99
        s.pv = pv
        s.g_hist = [] if keep_history else None
        s.load_hist = [] if keep_history else None
        s.next_jid = max(j.jid for j in s.jobs) + 1
        return s

    def clone(self, pv=None, events=None, bd_seed=None, p_act=None) -> 'State':
        """롤아웃용 복제. pv·events를 넘기면 그 미래로 바꿔 끼운다 (실제 미래는 보이지 않게).
        bd_seed를 주면 고장 난수도 새로 정함 (같은 seed = 같은 고장 미래 → 후보끼리 공정 비교)."""
        s = State.__new__(State)
        s.bd = self.bd
        s.bd_rng = np.random.default_rng(bd_seed) if bd_seed is not None else None
        s.ton = list(self.ton)
        s.bd_count, s.bd_slots = self.bd_count, self.bd_slots
        s.soc, s.g_target, s.b_plan, s.bat_mode, s.bat_kwh = self.soc, self.g_target, self.b_plan, self.bat_mode, self.bat_kwh
        s.p_act = self.p_act if p_act is None else p_act
        s.op_elapsed = list(self.op_elapsed)
        s.t = self.t
        s.ops = list(self.ops)
        s.jobs = list(self.jobs)
        s.op_status = list(self.op_status)
        s.op_m = list(self.op_m)
        s.op_rem = list(self.op_rem)
        s.op_end = list(self.op_end)
        s.m_state = list(self.m_state)
        s.m_warm = list(self.m_warm)
        s.m_op = list(self.m_op)
        s.m_down_until = list(self.m_down_until)
        s.m_idle_run = list(self.m_idle_run)
        s.m_startup_now = list(self.m_startup_now)
        s.queues = [list(q) for q in self.queues]
        s.pstart = dict(self.pstart)
        s.u_plan = self.u_plan
        s.unassigned = list(self.unassigned)
        s.energy, s.P_day = self.energy, self.P_day
        s.labor_used = set(self.labor_used)
        s.startups, s.pv_used, s.grid_kwh = self.startups, self.pv_used, self.grid_kwh
        if events is None:
            s.events, s.ev_idx = self.events, self.ev_idx
        else:
            s.events, s.ev_idx = sorted(events, key=lambda e: e[0]), 0
        s.last_event = self.last_event
        s.pv = self.pv if pv is None else pv
        s.g_hist = None
        s.load_hist = None
        s.next_jid = self.next_jid
        return s


# ---------------------------------------------------------------- 조회 도우미

def proc_time(st: State, o: int, m: int) -> int:
    return st.op_rem[o] if st.op_rem[o] > 0 else st.ops[o].proc[m][0]


def pred_done(st: State, o: int) -> bool:
    op = st.ops[o]
    if op.seq == 0:
        return st.t >= st.jobs[op.job].release
    return st.op_status[st.jobs[op.job].ops[op.seq - 1]] == DONE


def eligible(st: State, o: int, m: int) -> bool:
    return m in st.ops[o].proc and st.op_m[o] in (-1, m)


def can_start(st: State, inst: Instance, o: int, m: int) -> bool:
    lead = inst.machines[m].warmup if st.m_state[m] == OFF else 0
    return inst.labor.window_ok(st.t + lead, proc_time(st, o, m))


def ready_ops(st: State, inst: Instance, m: int) -> list:
    return [o for o in range(len(st.ops))
            if st.op_status[o] == WAIT and eligible(st, o, m) and pred_done(st, o) and can_start(st, inst, o, m)]


def remaining_work(st: State, o: int) -> int:
    """공정 o부터 작업 끝까지 남은 최소 처리시간."""
    op = st.ops[o]
    job = st.jobs[op.job]
    total = 0
    for oid in job.ops[op.seq:]:
        if st.op_status[oid] == DONE:
            continue
        total += st.op_rem[oid] if st.op_rem[oid] > 0 else min(p for p, _ in st.ops[oid].proc.values())
    return total


def slack(st: State, o: int) -> int:
    return st.jobs[st.ops[o].job].due - st.t - remaining_work(st, o)


def current_load(st: State, inst: Instance) -> float:
    load = inst.base_load[st.t]
    for m, spec in enumerate(inst.machines):
        s = st.m_state[m]
        if s == BUSY:
            load += st.ops[st.m_op[m]].proc[m][1]
        elif s in (IDLE, WARMING):
            load += spec.p_idle
    return load


# ---------------------------------------------------------------- 이벤트·행동·진행

def apply_events(st: State, inst: Instance):
    while st.ev_idx < len(st.events) and st.events[st.ev_idx][0] <= st.t:
        _, kind, payload = st.events[st.ev_idx]
        st.ev_idx += 1
        st.last_event = st.t
        if kind == 'urgent':
            job, ops = payload(st.next_jid, len(st.ops))
            st.next_jid += 1
            st.jobs.append(job)
            st.ops.extend(ops)
            for _ in ops:
                st.op_status.append(WAIT)
                st.op_m.append(-1)
                st.op_rem.append(-1)
                st.op_end.append(-1)
                st.op_elapsed.append(0)
            st.unassigned.extend(o.oid for o in ops)
        elif kind == 'breakdown':
            _break(st, *payload)
    _random_breakdowns(st)


def _random_breakdowns(st: State):
    """가동 중인 설비마다: 고장 확률 = 1 − exp(−θ_on × 누적 가동 칸), 수리는 칸마다 1 − exp(−θ_bd × 누적 고장 칸) 확률로 끝남."""
    if st.bd is None or st.bd_rng is None:
        return
    th_on, th_bd = st.bd
    for m in range(len(st.m_state)):
        if st.m_state[m] != BUSY:
            continue
        st.ton[m] += 1
        if st.bd_rng.random() < 1 - np.exp(-th_on * st.ton[m]):
            dur, k = 1, 1
            while st.bd_rng.random() >= 1 - np.exp(-th_bd * k):
                dur += 1
                k += 1
            st.ton[m] = 0
            st.bd_count += 1
            st.bd_slots += dur
            st.last_event = st.t
            _break(st, m, dur)


def _break(st: State, m: int, dur: int):
    if True:
            o = st.m_op[m]
            if o >= 0:
                st.op_status[o] = WAIT
                if st.m_state[m] == BUSY or st.op_rem[o] < st.ops[o].proc[m][0]:
                    # 가공 중단: 남은 시간 유지, 이 설비에 묶어 수리 후 맨 앞에서 재개
                    st.queues[m].insert(0, o)
                    st.pstart[o] = st.t
                else:
                    # 예열 중이던 예약 취소: 다시 배정 대상으로
                    st.op_m[o] = -1
                    st.op_rem[o] = -1
                    st.unassigned.append(o)
            st.m_state[m] = DOWN
            st.m_op[m] = -1
            st.m_warm[m] = 0
            st.m_down_until[m] = st.t + dur


def apply_action(st: State, inst: Instance, m: int, act: tuple):
    kind = act[0]
    s = st.m_state[m]
    if kind == 'start' and s in (IDLE, OFF):
        o = act[1]
        if st.op_status[o] != WAIT:
            return
        for q in st.queues:
            if o in q:
                q.remove(o)
        if o in st.unassigned:
            st.unassigned.remove(o)
        st.op_m[o] = m
        if st.op_rem[o] <= 0:
            st.op_rem[o] = st.ops[o].proc[m][0]
        st.m_op[m] = o
        if s == IDLE:
            st.m_state[m] = BUSY
            st.op_status[o] = RUN
        elif inst.machines[m].warmup == 0:          # 예열 없는 기계: 켜자마자 가공
            st.m_state[m] = BUSY
            st.m_startup_now[m] = True
            st.startups += 1
            st.op_status[o] = RUN
        else:
            st.m_state[m] = WARMING
            st.m_warm[m] = inst.machines[m].warmup
            st.m_startup_now[m] = True
            st.startups += 1
            st.op_status[o] = RESERVED
    elif kind == 'off' and s == IDLE:
        st.m_state[m] = OFF
    elif kind == 'on' and s == OFF and inst.machines[m].warmup == 0:
        st.m_state[m] = IDLE
        st.m_idle_run[m] = 0
        st.m_startup_now[m] = True
        st.startups += 1
    elif kind == 'on' and s == OFF:
        st.m_state[m] = WARMING
        st.m_warm[m] = inst.machines[m].warmup
        st.m_startup_now[m] = True
        st.startups += 1


def account_slot(st: State, inst: Instance):
    t = st.t
    load = inst.base_load[t]
    busy_any = False
    for m, spec in enumerate(inst.machines):
        s = st.m_state[m]
        if s == BUSY:
            load += st.ops[st.m_op[m]].proc[m][1]
            busy_any = True
        elif s in (IDLE, WARMING):
            load += spec.p_idle
        if st.m_startup_now[m]:
            load += spec.p_startup
            st.m_startup_now[m] = False
    pv = st.pv[t]
    net = load - pv
    pb = 0.0
    bat = inst.battery
    if bat:
        mode = st.bat_mode or 'greedy'
        if mode == 'follow' and st.b_plan is not None:
            want = float(st.b_plan[t])
        elif mode == 'track' and st.g_target is not None:
            want = net - float(st.g_target[t])
        elif mode == 'none':
            want = 0.0
        else:   # greedy: 남는 태양광은 충전, 비싼 시간에는 방전
            want = net if (net < 0 or inst.tariff.energy[t] > inst.tariff.energy.min()) else 0.0
        h, eta = I.SLOT_H, bat['eta']
        dis_max = min(bat['p_max'], max(0.0, st.soc - bat['soc_min']) * eta / h)
        ch_max = min(bat['p_max'], max(0.0, bat['soc_max'] - st.soc) / (eta * h))
        pb = min(max(want, -ch_max), dis_max)
        st.soc += (-pb * eta * h) if pb < 0 else (-pb * h / eta)
        st.bat_kwh += abs(pb) * h
        net -= pb
    if not bat and inst.feed_in == 0 and inst.tariff.energy[t] < 0:
        net = load          # 가격이 음수면 태양광을 끄고 전력망 전기를 씀 (받을수록 이득)
    g = max(0.0, net)
    st.energy += (inst.tariff.energy[t] * g - inst.feed_in * max(0.0, -net)) * I.SLOT_H
    if bat:
        st.energy += bat['deg'] * abs(pb) * I.SLOT_H
    st.grid_kwh += g * I.SLOT_H
    st.pv_used += min(pv, load) * I.SLOT_H
    st.P_day = max(st.P_day, g)
    if busy_any and not inst.labor.regular[t]:
        k = inst.labor.block_of[t]
        if k >= 0:
            st.labor_used.add(int(k))
    if st.g_hist is not None:
        st.g_hist.append(g)
        st.load_hist.append(load)


def advance(st: State, inst: Instance):
    for m in range(len(inst.machines)):
        s = st.m_state[m]
        if s == BUSY:
            o = st.m_op[m]
            st.op_rem[o] -= 1
            st.op_elapsed[o] += 1
            actual = st.p_act.get(o)
            if actual is not None and st.op_elapsed[o] < actual and st.op_rem[o] <= 0:
                st.op_rem[o] = 1           # 계획보다 길어짐: 최소 한 칸 더 남았다고 봄
            elif actual is not None and st.op_elapsed[o] >= actual:
                st.op_rem[o] = 0           # 계획보다 빨리 끝남
            if st.op_rem[o] == 0:
                st.op_status[o] = DONE
                st.op_end[o] = st.t + 1
                st.m_state[m] = IDLE
                st.m_op[m] = -1
                st.m_idle_run[m] = 0
        elif s == WARMING:
            st.m_warm[m] -= 1
            if st.m_warm[m] == 0:
                o = st.m_op[m]
                if o >= 0:
                    st.m_state[m] = BUSY
                    st.op_status[o] = RUN
                else:
                    st.m_state[m] = IDLE
                    st.m_idle_run[m] = 0
        elif s == IDLE:
            st.m_idle_run[m] += 1
        elif s == DOWN and st.t + 1 >= st.m_down_until[m]:
            st.m_state[m] = OFF
    st.t += 1


def free_machines(st: State) -> list:
    return [m for m, s in enumerate(st.m_state) if s in (IDLE, OFF)]


def step(st: State, inst: Instance, actions: dict):
    for m, a in actions.items():
        apply_action(st, inst, m, a)
    account_slot(st, inst)
    advance(st, inst)


def run(st: State, inst: Instance, policy, until: int | None = None, timer=None):
    until = I.SLOTS if until is None else until
    import time
    while st.t < until:
        apply_events(st, inst)
        t0 = time.perf_counter()
        acts = policy.decide(st, inst)
        if timer is not None:
            timer.append(time.perf_counter() - t0)
        step(st, inst, acts)
    return st


# ---------------------------------------------------------------- 비용

def cost_breakdown(st: State, inst: Instance) -> dict:
    tard = 0.0
    unfinished = 0
    for job in st.jobs:
        last = job.ops[-1]
        if st.op_status[last] == DONE:
            C = st.op_end[last]
        else:
            unfinished += 1
            C = I.SLOTS + sum(max(st.op_rem[o], 0) if st.op_rem[o] > 0 else
                            min(p for p, _ in st.ops[o].proc.values())
                            for o in job.ops if st.op_status[o] != DONE)
        tard += job.weight * max(0, C - job.due)
    ratchet = inst.tariff.demand_rate * inst.tariff.K * max(0.0, st.P_day - inst.P_ref)
    labor = sum(inst.labor.blocks[k][2] for k in st.labor_used)
    tardy_cost = inst.lam_tardy * tard
    unf = inst.unfinished_penalty * unfinished
    soc_adj = 0.0
    if inst.battery:   # 하루 끝 충전량이 시작보다 적으면 그만큼 싼 시간 전기로 다시 채우는 비용을 더함 (논문: 시작=끝)
        soc_adj = max(0.0, inst.battery['soc0'] - st.soc) * float(inst.tariff.energy.min()) / inst.battery['eta']
    energy = st.energy + soc_adj
    total = energy + ratchet + labor + tardy_cost + unf
    return dict(total=total, energy=energy, soc_end=st.soc, bat_kwh=st.bat_kwh, ratchet=ratchet, labor=labor,
                tardiness=tardy_cost, unfinished_pen=unf, unfinished=unfinished,
                peak_kw=st.P_day, startups=st.startups, grid_kwh=st.grid_kwh,
                pv_self_use=st.pv_used, co2_kg=st.grid_kwh * inst.emission_factor)


def total_cost(st: State, inst: Instance) -> float:
    return cost_breakdown(st, inst)['total']
