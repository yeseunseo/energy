"""기준 논문: Dunke & Nickel (2025) "Approximate and exact approaches to energy-aware job shop scheduling with
dynamic energy tariffs and power purchase agreements", Applied Energy 380, 125065 (CC BY).
공개 코드(gitlab.kit.edu/fabian.dunke/energyAwareJobShopScheduling)의 데이터 생성 · 순서 규칙을 그대로 옮김.

논문 설정
  - 기계 3대 (seed 0): 가동 전력 1~5, 대기 = 가동의 25%
  - 작업 3·5·10개 (seed = 날짜 번호 × 작업 수, 시작 08:00 기준 하루 안에 못 끝나면 seed + 1)
  - 하루 96칸(15분), 가격 = 2023년 독일 하루 전 시장 가격 (시간 가격을 4칸에 나눠 담음, 서머타임 없는 UTC+1)
  - 기계는 처음 켠 뒤 마지막 작업이 끝날 때까지 꺼지지 않음 (중간은 대기)
  - 비용 = Σ 가격 × 사용량 (원 논문 단위: €/MWh × 전력 단위, 시간 곱 없음)
"""
from __future__ import annotations

import math
import random

import numpy as np
import pandas as pd

CALENDAR_DAYS = [15, 46, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349]
MAX_P = {3: 20, 5: 15, 10: 10, 20: 8}   # 20: 확장 실험용 (논문 밖)
PPA_PRICE = 95.18


# ---------------------------------------------------------------- 가격
def load_prices(xlsx_path: str, n_days: int = 365) -> dict:
    """논문 getHistoricalEnergyData와 같은 방식."""
    df = pd.read_excel(xlsx_path)
    prices = {d: [-9999.0] * 96 for d in range(1, n_days + 1)}
    counter = -4
    for _, row in df.iterrows():
        if not (isinstance(row['Tag'], float) and math.isnan(row['Tag'])):
            day = int(row['Tag'])
            p = float(row['Day Ahead Auktion (DE-LU)'])
            for k in range(4):
                prices[day][counter + k] = p
        counter += 4
        if counter == 96:
            counter = 0
    return prices


def effective_price(day_prices: list, fixed: str, ppa: str) -> np.ndarray:
    """두 공급원(기존·재생) 중 칸마다 싼 쪽을 고르는 논문 모형과 같음 (사용량 ≥ 0이므로 칸별 최솟값)."""
    conv = np.array(day_prices, dtype=float)
    renew = conv.copy()
    if ppa == 'ppa':
        renew[:] = PPA_PRICE
    if fixed == 'fix':
        conv[:] = conv.mean()
    return np.minimum(conv, renew)


# ---------------------------------------------------------------- 기계 · 작업
def generate_machines(n: int = 3) -> dict:
    random.seed(0)
    machines = {}
    for i in range(n):
        kw = random.randint(1, 5)
        em = random.randint(1, 5)
        machines[f'M{i + 1}'] = {'consumption': {'on': kw, 'idle': 0.25 * kw}, 'emissions': {'on': em, 'idle': 0.25 * em}}
    return machines


def generate_jobs(num_jobs: int, machines: dict, seed: int) -> dict:
    random.seed(seed)
    keys = list(machines.keys())
    pmax = MAX_P[num_jobs]
    jobs = {}
    for j in range(1, num_jobs + 1):
        seq = random.sample(keys, k=random.randint(2, len(keys)))
        times = {m: random.randint(2, pmax) for m in seq}
        jobs[f'J{j}'] = {'sequence': seq, 'machine times': times}
    for m in machines:
        if not any(m in jobs[j]['sequence'] for j in jobs):
            first = list(jobs)[0]
            jobs[first]['sequence'].append(m)
            jobs[first]['machine times'][m] = random.randint(2, pmax)
    return jobs


def operations(jobs: dict):
    ops, precs = [], []
    for j in jobs:
        for m in jobs[j]['sequence']:
            ops.append({'machine': m, 'job': j, 'processing time': jobs[j]['machine times'][m]})
            if jobs[j]['sequence'].index(m) > 0:
                precs.append({'from': ops[-2], 'to': ops[-1]})
    return ops, precs


# ---------------------------------------------------------------- 가장 많이 남은 일 순서 규칙 (논문 mostRemainingWork.py)
def mrw_sequences(machines, ops, precs) -> dict:
    nodes = [(o['machine'], o['job']) for o in ops]
    edges = [{'start': (p['from']['machine'], p['from']['job']), 'end': (p['to']['machine'], p['to']['job']),
              'length': p['from']['processing time']} for p in precs]
    starts = [n for n in nodes if not any(e['end'] == n for e in edges)]
    ends = [n for n in nodes if not any(e['start'] == n for e in edges)]
    for n in starts:
        edges.append({'start': 'source', 'end': n, 'length': 0})
    for n in ends:
        p = next(o['processing time'] for o in ops if (o['machine'], o['job']) == n)
        edges.append({'start': n, 'end': 'sink', 'length': p})
    unscheduled = list(nodes)
    total = len(nodes)
    scheduled = []
    cand = [e['end'] for e in edges if e['start'] == 'source']
    while len(scheduled) < total:
        rw = {m: sum(e['length'] for e in edges if e['start'] in unscheduled and e['start'][0] == m) for m in machines}
        best_m, best = '', 0
        for n in cand:
            if rw[n[0]] > best:
                best, best_m = rw[n[0]], n[0]
        nxt = next(n for n in cand if n[0] == best_m)
        cand.remove(nxt)
        unscheduled.remove(nxt)
        succ = next((e['end'] for e in edges if e['start'] == nxt), None)
        if not isinstance(succ, str) and succ is not None:
            cand.append(succ)
        scheduled.append(nxt)
    seqs = {}
    for m in machines:
        seqs[m] = [n[1] for n in scheduled if n[0] == m]
    return seqs


def earliest_schedule_fits(jobs, ops, seqs, start_slot: int) -> bool:
    """순서를 고정하고 가장 이르게 배치했을 때 하루(96칸) 안에 끝나는지 (논문 생성 단계의 '풀 수 있음' 검사)."""
    end = {}
    pending = [(o['machine'], o['job']) for o in ops]
    pos = {m: 0 for m in seqs}
    mfree = {m: start_slot for m in seqs}
    guard = 0
    while pending and guard < 10_000:
        guard += 1
        progressed = False
        for m in seqs:
            if pos[m] >= len(seqs[m]):
                continue
            j = seqs[m][pos[m]]
            seq = jobs[j]['sequence']
            k = seq.index(m)
            if k > 0 and (seq[k - 1], j) not in end:
                continue
            s = max(mfree[m], end[(seq[k - 1], j)] if k > 0 else start_slot)
            e = s + jobs[j]['machine times'][m]
            end[(m, j)] = e
            mfree[m] = e
            pos[m] += 1
            pending.remove((m, j))
            progressed = True
        if not progressed:
            return False
    return not pending and max(end.values()) <= 96


def generate_instance_data(num_jobs: int, calendar_day: int, n_machines: int = 3):
    machines = generate_machines(n_machines)
    seed = calendar_day * num_jobs
    while True:
        jobs = generate_jobs(num_jobs, machines, seed)
        ops, precs = operations(jobs)
        seqs = mrw_sequences(machines, ops, precs)
        if earliest_schedule_fits(jobs, ops, seqs, 32):
            return machines, jobs, ops, precs, seqs, seed
        seed += 1


# ---------------------------------------------------------------- 우리 계획 모형용 인스턴스
def build_instance(machines, jobs, ops, seqs, price: np.ndarray, start_slot: int, alg: str):
    """alg='ex': 순서까지 최적화 / 'heu': 기계별 작업 순서를 순서 규칙(MRW)으로 고정 (논문 근사 방식)."""
    from . import instance as I
    from .instance import MachineSpec, OpSpec, JobSpec, Tariff, Labor, Instance
    assert I.SLOT_MIN == 15
    mnames = list(machines)
    mspecs = [MachineSpec(m, m, machines[m]['consumption']['idle'], 0.0, 0, float(machines[m]['consumption']['on'])) for m in mnames]
    op_list, job_list, oid_of = [], [], {}
    for jn, j in enumerate(jobs):
        oids = []
        for k, m in enumerate(jobs[j]['sequence']):
            oid = len(op_list)
            mi = mnames.index(m)
            op_list.append(OpSpec(oid, jn, k, m, {mi: (jobs[j]['machine times'][m], float(machines[m]['consumption']['on']))}))
            oid_of[(m, j)] = oid
            oids.append(oid)
        job_list.append(JobSpec(jn, I.SLOTS, 1, tuple(oids), release=start_slot))
    prec = None
    if alg == 'heu':
        prec = [(oid_of[(m, a)], oid_of[(m, b)]) for m in seqs for a, b in zip(seqs[m], seqs[m][1:])]
    tariff = Tariff(energy=np.round(price * 100).astype(int), demand_rate=0, K=0, season='dunke')
    labor = Labor(regular=np.ones(I.SLOTS, dtype=bool), blocks=[])
    return Instance(machines=mspecs, ops=op_list, jobs=job_list, tariff=tariff, labor=labor,
                    base_load=np.zeros(I.SLOTS), pv_cap=0.0, P_bar=1e9, month_max=0.0,
                    lam_tardy=0, unfinished_penalty=10_000_000, extra_prec=prec, single_on=True)


def paper_cost(plan_objective: float) -> float:
    """우리 목적값(가격×100 × 사용량 × 0.25시간) → 논문 단위(가격 × 사용량)."""
    return plan_objective / 25.0


# ---------------------------------------------------------------- ② 불확실성 실험용
PV_CAP_RATIO = 0.5    # 공장 태양광 최대 출력 ≈ 공장 최대 부하(모든 기계 가동)의 절반


def urgent_maker(machines: dict, num_jobs: int):
    """급주문: 논문 작업 생성 규칙과 같은 방식(기계 2~3대, 처리 시간 2~최대)으로 하루 중간에 들어오는 작업."""
    from .instance import OpSpec, JobSpec
    from . import instance as I
    mnames = list(machines)
    pmax = MAX_P[num_jobs]

    def maker(rng, slot):
        seed = int(rng.integers(1 << 30))

        def payload(jid, first_oid):
            r = np.random.default_rng(seed)
            k = int(r.integers(2, len(mnames) + 1))
            seq = list(r.permutation(mnames)[:k])
            ops, oid = [], first_oid
            for s, m in enumerate(seq):
                p = int(r.integers(2, pmax + 1))
                mi = mnames.index(m)
                ops.append(OpSpec(oid, jid, s, m, {mi: (p, float(machines[m]['consumption']['on']))}))
                oid += 1
            return JobSpec(jid, I.SLOTS, 1, tuple(o.oid for o in ops), release=slot, urgent=True), ops
        return payload
    return maker


def sample_actual_times(ops, rng, spread: float = 0.2) -> dict:
    """실제 처리 칸 = 계획 × U(1−spread, 1+spread), 반올림, 최소 1칸."""
    return {o.oid: max(1, int(round(list(o.proc.values())[0][0] * rng.uniform(1 - spread, 1 + spread)))) for o in ops}


def earliest_plan(inst, jobs: dict, seqs: dict, start_slot: int):
    """기존 운영: 순서 규칙(MRW)으로 정한 기계별 순서대로, 가격을 보지 않고 가장 이르게 시작하는 계획."""
    from . import instance as I
    from .planner import Plan
    mnames = [m.name for m in inst.machines]
    oid_of = {}
    for jn, j in enumerate(jobs):
        for k, m in enumerate(jobs[j]['sequence']):
            oid_of[(m, j)] = inst.jobs[jn].ops[k]
    queues = [[oid_of[(m, j)] for j in seqs[m]] for m in mnames]
    pstart, end = {}, {}
    mfree = {mi: start_slot for mi in range(len(mnames))}
    pos = [0] * len(mnames)
    remaining = sum(len(q) for q in queues)
    while remaining:
        for mi, q in enumerate(queues):
            if pos[mi] >= len(q):
                continue
            o = q[pos[mi]]
            op = inst.ops[o]
            job = inst.jobs[op.job]
            prev = job.ops[op.seq - 1] if op.seq > 0 else None
            if prev is not None and prev not in end:
                continue
            p = op.proc[mi][0]
            s = max(mfree[mi], end[prev] if prev is not None else start_slot)
            pstart[o], end[o] = s, s + p
            mfree[mi] = s + p
            pos[mi] += 1
            remaining -= 1
    u = []
    for mi, q in enumerate(queues):
        a = np.zeros(I.SLOTS, dtype=int)
        if q:
            a[min(pstart[o] for o in q):max(end[o] for o in q)] = 1
        u.append(a)
    return Plan(queues, pstart, u, [], 'EARLIEST', float('nan'), float('nan'), 0.0, [])
