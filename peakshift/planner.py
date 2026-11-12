"""하루 전 계획(및 MPC 재계획)용 CP-SAT 시간 색인 모형.

목적함수 (모든 항을 원 × 40 단위의 정수로 만든다. 전력은 0.1 kW 단위 정수):
  Σ c_t g_t                     전력량 요금      (c 원/kWh × g 0.1kW × 0.25h = c·g/40 원)
  + 4 ρ K E                     기본요금 증가분  (E = max(0, P − P̄), 0.1kW 단위)
  + 40 Σ w_k z_k                가산 인건비
  + 40 λ Σ w_j T_j              납기 지연
  + 40 · 미완료 벌점 · Σ skip     당일 배정 불가 공정
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from ortools.sat.python import cp_model

from . import instance as I
from .instance import Instance
from . import sim as S

SCALE = 10  # kW -> 0.1kW 정수


@dataclass
class Plan:
    queues: list          # 설비별 공정 순서
    pstart: dict          # 공정 -> 계획 시작 슬롯 (가공 시작 기준)
    u: list               # 설비별 켜짐 계획 (길이 I.SLOTS, 0/1)
    z: list               # 근무 블록 편성 여부
    status: str
    objective: float      # 원
    bound: float          # 원
    wall: float
    skipped: list
    n_vars: int = 0
    n_cons: int = 0
    g_plan: np.ndarray | None = None    # 계획한 한전 전력 kW (칸별)
    b_plan: np.ndarray | None = None    # 계획한 배터리 출력 kW (+방전 / −충전)


def _i(x):
    return int(round(x * SCALE))


def solve(inst: Instance, pv_fc: np.ndarray, st: S.State | None = None,
          time_limit: float = 30.0, workers: int = 1, hint: dict | None = None) -> Plan:
    t0 = st.t if st is not None else 0
    M = len(inst.machines)
    if st is None:
        st = S.State.initial(inst, pv_fc, [], keep_history=False)
    ops, jobs = st.ops, st.jobs
    mdl = cp_model.CpModel()
    T = range(t0, I.SLOTS)

    # ---- 고정 부하: 가공 중(RUN)·예열 예약(RESERVED) 공정
    fixed_busy = [np.zeros(I.SLOTS, dtype=bool) for _ in range(M)]
    fixed_extra = np.zeros(I.SLOTS)            # 고정 가공의 (가공 − 대기) 전력
    finish_known = {}                        # oid -> 완료 슬롯
    forced_on = [np.zeros(I.SLOTS, dtype=bool) for _ in range(M)]
    blocked = [np.zeros(I.SLOTS, dtype=bool) for _ in range(M)]   # 가공 불가 슬롯 (예열·고장)
    for m in range(M):
        s = st.m_state[m]
        if s == S.BUSY:
            o = st.m_op[m]
            end = min(I.SLOTS, t0 + st.op_rem[o])
            fixed_busy[m][t0:end] = True
            forced_on[m][t0:end] = True
            fixed_extra[t0:end] += ops[o].proc[m][1] - inst.machines[m].p_idle
            finish_known[o] = end
        elif s == S.WARMING:
            w_end = t0 + st.m_warm[m]
            forced_on[m][t0:w_end] = True
            blocked[m][t0:w_end] = True
            o = st.m_op[m]
            if o >= 0:
                end = min(I.SLOTS, w_end + st.op_rem[o])
                fixed_busy[m][w_end:end] = True
                forced_on[m][w_end:end] = True
                fixed_extra[w_end:end] += ops[o].proc[m][1] - inst.machines[m].p_idle
                finish_known[o] = end
        elif s == S.DOWN:
            blocked[m][t0:min(I.SLOTS, st.m_down_until[m])] = True
    for o in range(len(ops)):
        if st.op_status[o] == S.DONE:
            finish_known[o] = st.op_end[o]

    # ---- 시작 변수 x[o,m,s]
    todo = [o for o in range(len(ops)) if st.op_status[o] == S.WAIT]
    x = {}
    busy_terms = [[[] for _ in range(I.SLOTS)] for _ in range(M)]
    load_terms = [([], []) for _ in range(I.SLOTS)]
    start_expr, end_expr, skip = {}, {}, {}
    for o in todo:
        op = ops[o]
        job = jobs[op.job]
        est = max(t0, job.release)
        if op.seq > 0:
            prev = job.ops[op.seq - 1]
            if prev in finish_known:
                est = max(est, finish_known[prev])
        s_vars, s_coefs, e_coefs = [], [], []
        for m, (p_full, kw) in op.proc.items():
            if st.op_m[o] not in (-1, m):
                continue
            p = st.op_rem[o] if st.op_rem[o] > 0 else p_full
            extra = _i(kw) - _i(inst.machines[m].p_idle)
            lo, hi = est, I.SLOTS - p
            if inst.op_window and o < len(inst.ops) and o in inst.op_window:
                w0, w1 = inst.op_window[o]
                lo, hi = max(lo, w0), min(hi, w1 + (p_full - p))
            for s in range(lo, hi + 1):
                if not inst.labor.window_ok(s, p) or blocked[m][s:s + p].any() or fixed_busy[m][s:s + p].any():
                    continue
                v = mdl.NewBoolVar(f'x{o}_{m}_{s}')
                x[o, m, s] = v
                s_vars.append(v)
                s_coefs.append(s)
                e_coefs.append(s + p)
                for t in range(s, s + p):
                    busy_terms[m][t].append(v)
                    load_terms[t][0].append(v)
                    load_terms[t][1].append(extra)
        sk = mdl.NewBoolVar(f'skip{o}')
        skip[o] = sk
        mdl.Add(sum(s_vars) + sk == 1)
        start_expr[o] = cp_model.LinearExpr.WeightedSum(s_vars + [sk], s_coefs + [I.SLOTS])
        end_expr[o] = cp_model.LinearExpr.WeightedSum(s_vars + [sk], e_coefs + [2 * I.SLOTS])

    # 추가 순서 (같은 작업끼리 번호 순서 등)
    for a, b in (inst.extra_prec or []):
        if a in end_expr and b in start_expr:
            mdl.Add(start_expr[b] >= end_expr[a])

    # 선후행
    for o in todo:
        op = ops[o]
        if op.seq > 0:
            prev = jobs[op.job].ops[op.seq - 1]
            if prev in start_expr:
                mdl.Add(start_expr[o] >= end_expr[prev])
                mdl.AddImplication(skip[prev], skip[o])

    # ---- 설비 상태 u(켜짐), v(기동)
    u = [[None] * I.SLOTS for _ in range(M)]
    v = [[None] * I.SLOTS for _ in range(M)]
    simple = [sp.p_idle == 0 and sp.p_startup == 0 and sp.warmup == 0 for sp in inst.machines]
    for m, spec in enumerate(inst.machines):
        if simple[m]:
            # 대기 전력·기동 전력·예열이 없는 기계(논문 공장): 켜짐 변수 없이 '한 번에 하나'만 지키면 됨
            for t in T:
                if busy_terms[m][t]:
                    mdl.Add(sum(busy_terms[m][t]) + int(fixed_busy[m][t]) <= 1)
            continue
        init_on = 1 if st.m_state[m] in (S.IDLE, S.BUSY, S.WARMING) else 0
        for t in T:
            u[m][t] = mdl.NewBoolVar(f'u{m}_{t}')
            v[m][t] = mdl.NewBoolVar(f'v{m}_{t}')
            if forced_on[m][t]:
                mdl.Add(u[m][t] == 1)
            if st.m_state[m] == S.DOWN and t < st.m_down_until[m]:
                mdl.Add(u[m][t] == 0)
            mdl.Add(sum(busy_terms[m][t]) + int(fixed_busy[m][t]) <= u[m][t])
            prev_u = u[m][t - 1] if t > t0 else init_on
            mdl.Add(v[m][t] >= u[m][t] - prev_u)
            if st.m_state[m] == S.WARMING and t < t0 + st.m_warm[m]:
                mdl.Add(v[m][t] == 0)
        if inst.single_on and t0 == 0:
            mdl.Add(sum(v[m][t] for t in T) <= 1)
        # 예열: 기동 후 warmup 슬롯 동안 가공 불가, 켜져 있어야 함
        for t in T:
            for tt in range(t, min(I.SLOTS, t + spec.warmup)):
                mdl.Add(sum(busy_terms[m][tt]) + int(fixed_busy[m][tt]) + v[m][t] <= 1)
                mdl.Add(u[m][tt] >= v[m][t])

    # ---- 근무 블록
    z = []
    labor_cost = []
    for k, (a, b, cost) in enumerate(inst.labor.blocks):
        if k in st.labor_used:
            z.append(1)
            continue
        if b <= t0:
            z.append(0)
            continue
        zk = mdl.NewBoolVar(f'z{k}')
        z.append(zk)
        labor_cost.append((zk, cost))
        for t in range(max(a, t0), b):
            for m in range(M):
                if busy_terms[m][t]:
                    mdl.Add(sum(busy_terms[m][t]) <= zk)

    MULT = int(round(SCALE / I.SLOT_H))   # 목적함수 정수 배율 (15분: 40, 5분: 120)
    fi = int(round(inst.feed_in))
    feed_terms = []
    exact_load = (not inst.battery) and fi == 0 and float(np.max(pv_fc[t0:])) <= 0 if t0 < I.SLOTS else False
    # ---- 배터리 (선택): 충전 ch, 방전 dis (0.1kW), 충전량 soc (정수 배율 F: 방전 계수 100이 되도록)
    bat = inst.battery
    ch, dis, soc = {}, {}, {}
    if bat:
        h, eta = I.SLOT_H, bat['eta']
        F = 1000 * eta / h
        cc, cd = int(round(eta * h / SCALE * F)), int(round(h / (eta * SCALE) * F))
        pmx = _i(bat['p_max'])
        lo_s, hi_s = int(round(bat['soc_min'] * F)), int(round(bat['soc_max'] * F))
        soc[t0] = int(round(min(max(st.soc, bat['soc_min']), bat['soc_max']) * F))
        for t in T:
            ch[t] = mdl.NewIntVar(0, pmx, f'ch{t}')
            dis[t] = mdl.NewIntVar(0, pmx, f'dis{t}')
            bt = mdl.NewBoolVar(f'bm{t}')
            mdl.Add(ch[t] <= pmx * bt)
            mdl.Add(dis[t] <= pmx * (1 - bt))
            soc[t + 1] = mdl.NewIntVar(lo_s, hi_s, f'soc{t + 1}')
            mdl.Add(soc[t + 1] == soc[t] + cc * ch[t] - cd * dis[t])
        mdl.Add(soc[I.SLOTS] >= int(round(bat['soc0'] * F)))
    # ---- 전력과 피크
    gmax = _i(inst.base_load.max() + sum(sp.p_proc * 1.3 + sp.p_startup for sp in inst.machines))
    g = {}
    P = mdl.NewIntVar(0, gmax, 'P')
    mdl.Add(P >= _i(st.P_day))
    for t in T:
        terms, coefs = list(load_terms[t][0]), list(load_terms[t][1])
        for m, spec in enumerate(inst.machines):
            if simple[m]:
                continue
            terms += [u[m][t], v[m][t]]
            coefs += [_i(spec.p_idle), _i(spec.p_startup)]
        const = _i(inst.base_load[t]) + _i(fixed_extra[t]) - _i(pv_fc[t])
        if bat:
            terms += [ch[t], dis[t]]
            coefs += [1, -1]
        g[t] = mdl.NewIntVar(0, gmax, f'g{t}')
        if exact_load:   # 태양광·배터리·판매가 없으면 한전 전력 = 부하 (음수 가격일 때 g가 부풀지 않게)
            mdl.Add(g[t] == cp_model.LinearExpr.WeightedSum(terms, coefs) + const)
        if fi:
            feed_terms += [(vv, fi * cc) for vv, cc in zip(terms, coefs)]
        mdl.Add(g[t] >= cp_model.LinearExpr.WeightedSum(terms, coefs) + const)
        if not exact_load and not bat and not fi:   # 한전 전력은 부하를 넘을 수 없음 (음수 가격에서 g가 부풀지 않게)
            mdl.Add(g[t] <= cp_model.LinearExpr.WeightedSum(terms, coefs) + const + _i(pv_fc[t]))
        mdl.Add(P >= g[t])
    E = mdl.NewIntVar(0, gmax, 'E')
    mdl.Add(E >= P - _i(inst.P_ref))

    # ---- 납기 지연
    tard_terms = []
    for job in jobs:
        last = job.ops[-1]
        if last in end_expr:
            Tj = mdl.NewIntVar(0, 3 * I.SLOTS, f'T{job.jid}')
            mdl.Add(Tj >= end_expr[last] - job.due)
            tard_terms.append((Tj, job.weight))

    # ---- 목적함수 (원 × 40)
    obj = []
    # 남는 태양광 판매: 수입 = fi × (g − 부하 + 태양광) → g 계수는 (c − fi), 부하 항에 +fi (상수는 생략)
    obj += [(g[t], int(inst.tariff.energy[t]) - fi) for t in T]
    obj += feed_terms
    obj.append((E, (MULT // SCALE) * inst.tariff.demand_rate * inst.tariff.K))
    obj += [(zk, MULT * c) for zk, c in labor_cost]
    obj += [(Tj, MULT * inst.lam_tardy * w) for Tj, w in tard_terms]
    obj += [(sk, MULT * inst.unfinished_penalty) for sk in skip.values()]
    if bat:   # 배터리 열화 비용 (요금 단위/kWh, 최소 1로 반올림)
        dg = max(1, int(round(bat['deg'])))
        obj += [(ch[t], dg) for t in T] + [(dis[t], dg) for t in T]
    mdl.Minimize(cp_model.LinearExpr.WeightedSum([a for a, _ in obj], [b for _, b in obj]))

    if hint:
        for (o, m, s), var in x.items():
            mdl.AddHint(var, 1 if hint.get(o) == (m, s) else 0)

    proto = mdl.Proto()
    n_vars, n_cons = len(proto.variables), len(proto.constraints)
    solver = cp_model.CpSolver()
    # 재현성: 컴퓨터 속도와 상관없이 같은 답이 나오도록 '정해진 계산량(결정적 시간)'으로 멈춤.
    # 너무 느린 컴퓨터를 대비해 실제 시간은 3배까지만 허용.
    solver.parameters.max_deterministic_time = time_limit
    solver.parameters.max_time_in_seconds = time_limit * 3
    solver.parameters.num_workers = workers
    solver.parameters.random_seed = 0
    if workers > 1:
        solver.parameters.interleave_search = True   # 여러 스레드여도 결과가 같게
    res = solver.Solve(mdl)
    status = solver.StatusName(res)
    if res not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return Plan([[] for _ in range(M)], {}, [np.zeros(I.SLOTS, int) for _ in range(M)],
                    [0] * len(inst.labor.blocks), status, float("nan"), float("nan"), solver.WallTime(), todo, n_vars, n_cons)

    starts = {}
    for (o, m, s), var in x.items():
        if solver.Value(var):
            starts[o] = (m, s)
    queues = [[] for _ in range(M)]
    for o, (m, s) in sorted(starts.items(), key=lambda kv: kv[1][1]):
        queues[m].append(o)
    pstart = {o: s for o, (m, s) in starts.items()}
    u_arr = []
    for m in range(M):
        a = np.zeros(I.SLOTS, dtype=int)
        if simple[m]:
            for o, (mm, s0) in starts.items():
                if mm == m:
                    p = st.op_rem[o] if st.op_rem[o] > 0 else ops[o].proc[m][0]
                    a[s0:s0 + p] = 1
        else:
            for t in T:
                a[t] = solver.Value(u[m][t])
        u_arr.append(a)
    z_val = [zk if isinstance(zk, int) else solver.Value(zk) for zk in z]
    skipped = [o for o, sk in skip.items() if solver.Value(sk)]
    const = (fi * sum(_i(inst.base_load[t]) - _i(pv_fc[t]) for t in T)) if fi else 0   # 판매 수입 식의 상수항
    g_plan = np.zeros(I.SLOTS)
    b_plan = np.zeros(I.SLOTS)
    for t in T:
        g_plan[t] = solver.Value(g[t]) / SCALE
        if bat:
            b_plan[t] = (solver.Value(dis[t]) - solver.Value(ch[t])) / SCALE
    return Plan(queues, pstart, u_arr, z_val, status, (solver.ObjectiveValue() + const) / MULT,
                (solver.BestObjectiveBound() + const) / MULT, solver.WallTime(), skipped, n_vars, n_cons,
                g_plan=g_plan, b_plan=b_plan if bat else None)


def hint_from_run(st: S.State) -> dict:
    """시뮬레이션 결과(예: 납기 순 규칙)를 CP-SAT 초기해 힌트로 변환."""
    h = {}
    for o in range(len(st.ops)):
        if st.op_status[o] == S.DONE and st.op_m[o] >= 0:
            p = st.ops[o].proc[st.op_m[o]][0]
            h[o] = (st.op_m[o], st.op_end[o] - p)
    return h
