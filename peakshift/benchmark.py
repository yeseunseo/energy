"""기준 논문 공장: Shao, Li & Tan (2023) "Event-triggered Hybrid Energy-aware Scheduling in Manufacturing Systems"
(arXiv 2302.00812, 정식판 IJPR 2024 62(21)) 4.1절의 기계 10대 흐름 공정.

논문 그대로 가져온 것
  - 기계 10대 전력 θ1 (kW), 공정 시간 θ2 (5분 단위), 하루 작업 25개, 24시간 · 5분 단위(288칸)
  - 요금: 9~21시 0.330달러/kWh, 그 외 0.187달러/kWh, 남는 태양광 판매 0.052달러/kWh
  - 고장: 가동 중 칸마다 1 − exp(−0.002 × 누적 가동 칸) 확률로 고장, 수리는 1 − exp(−1 × 누적 고장 칸) 확률로 끝남
  - 기계는 켜짐(가공) / 꺼짐만 있음 → 대기 전력 0

바꾼 것 (논문 미공개 또는 범위 밖)
  - 태양광: 논문의 호주 일사량(날짜 미공개) 대신 구미 실제 발전 기록을 100kWp로 환산
  - 배터리: 논문과 같음 (50kWh, 방전 깊이 80% → 5~45kWh, 시작·끝 25kWh, 최대 50kW, 효율 0.9)
  - 가스터빈: 제외 (가스 가격 단위가 불분명)
금액 단위: 0.001달러 (정수 계산용). 결과 표에서 달러로 바꿈.
"""
from __future__ import annotations

import numpy as np

from . import instance as I
from .instance import MachineSpec, OpSpec, JobSpec, Tariff, Labor, Instance

POWER_KW = [50.63, 22.4, 5.12, 5.28, 12.68, 35.14, 6.58, 5.21, 8.4, 2.15]
PROC_5MIN = [5, 2, 8, 5, 3, 8, 4, 6, 6, 7]
N_JOBS = 25
PRICE_PEAK, PRICE_OFF, FEED_IN = 330, 187, 52     # 0.001달러/kWh
BREAKDOWN = (0.002, 1.0)                           # (θ12, θ13)


BATTERY = dict(cap=50.0, soc_min=5.0, soc_max=45.0, soc0=25.0, p_max=50.0, eta=0.9, deg=0.6)   # 논문 θ4~θ9, 열화 0.0006달러/kWh


def build_paper_instance(pv_cap: float = 100.0, idle_ratio: float = 0.0, battery: bool = True) -> Instance:
    """idle_ratio > 0이면 가공 안 할 때 켜져 있으면 가공 전력 × idle_ratio를 씀 (SPARK 실측 확장용)."""
    assert I.SLOT_MIN == 5, '논문 공장은 5분 단위: instance.set_resolution(5)를 먼저 호출'
    machines = [MachineSpec(f'M{i + 1}', f'stage{i + 1}', idle_ratio * p, 0.0, 0, p) for i, p in enumerate(POWER_KW)]
    ops, jobs = [], []
    for j in range(N_JOBS):
        oids = []
        for i in range(len(machines)):
            oid = len(ops)
            ops.append(OpSpec(oid, j, i, f'stage{i + 1}', {i: (PROC_5MIN[i], POWER_KW[i])}))
            oids.append(oid)
        jobs.append(JobSpec(j, I.SLOTS, 1, tuple(oids)))
    e = np.array([PRICE_PEAK if 9 <= t * I.SLOT_H < 21 else PRICE_OFF for t in range(I.SLOTS)], dtype=int)
    tariff = Tariff(energy=e, demand_rate=0, K=0, season='paper')
    labor = Labor(regular=np.ones(I.SLOTS, dtype=bool), blocks=[])      # 논문: 24시간 언제든 작업 가능
    # 25개 작업이 모두 같으므로 기계마다 작업 번호 순서로 처리해도 손해가 없음 → 순서를 고정해 모형을 줄임
    K, p = len(machines), PROC_5MIN
    est = [[0] * K for _ in range(N_JOBS)]
    for j in range(N_JOBS):
        for k in range(K):
            est[j][k] = max(est[j - 1][k] + p[k] if j else 0, est[j][k - 1] + p[k - 1] if k else 0)
    lst = [[0] * K for _ in range(N_JOBS)]
    for j in range(N_JOBS - 1, -1, -1):
        for k in range(K - 1, -1, -1):
            cand = [I.SLOTS - p[k]]
            if j + 1 < N_JOBS:
                cand.append(lst[j + 1][k] - p[k])
            if k + 1 < K:
                cand.append(lst[j][k + 1] - p[k])
            lst[j][k] = min(cand)
    window = {jobs[j].ops[k]: (est[j][k], lst[j][k]) for j in range(N_JOBS) for k in range(K)}
    prec = [(jobs[j].ops[k], jobs[j + 1].ops[k]) for j in range(N_JOBS - 1) for k in range(K)]
    return Instance(machines=machines, ops=ops, jobs=jobs, tariff=tariff, labor=labor,
                    base_load=np.zeros(I.SLOTS), pv_cap=pv_cap, P_bar=1e9, month_max=0.0,
                    lam_tardy=0, unfinished_penalty=1_000_000, feed_in=FEED_IN,
                    op_window=window, extra_prec=prec, battery=dict(BATTERY) if battery else None)
