"""③ 한국 적용: 같은 논문 공장(Dunke & Nickel 2025)에 한국 2026 요금 · 래칫 · 인건비를 넣는다.

요금 (산업용(을) 고압A 선택Ⅱ, 2026.4.16 개편 후, 원/kWh): 기본공급약관 별표1 기준 (biz-energy.org 칼럼, 정부 개편 발표와 교차 확인)
  여름(6~8월) 116.0 / 163.8 / 229.0 · 봄가을(3~5, 9~10월) 116.0 / 133.4 / 150.9 · 겨울(11~2월) 123.0 / 164.0 / 204.6
  기본요금 8,320원/kW
시간대
  3~10월(개편): 경부하 22~08시, 최대부하 15~21시, 나머지 중간부하
  11~2월(기존 유지): 경부하 22~08시, 최대부하 09~12·16~19시, 나머지 중간부하
래칫: 요금적용전력 = 직전 12개월 중 12·1·2·7·8·9월 및 당월 최대 → 그 달에 넘기면 12개월 유지(K=12), 그 외 달은 당월만(K=1)
인건비: 정규 08~17시. 연장 블록 06~08·17~19·19~21·21~24시, 연장 50% 가산, 22시 이후 야간 50% 추가 (근로기준법 제56조)
금액은 0.1원 단위 정수로 계산하고 결과는 원으로 바꿔 보고.
"""
from __future__ import annotations

import copy

import numpy as np

from . import instance as I
from .instance import Tariff, Labor
from . import dunke as D

PRICES = {'summer': (116.0, 163.8, 229.0), 'springfall': (116.0, 133.4, 150.9), 'winter': (123.0, 164.0, 204.6)}
DEMAND_RATE = 8320
KW_PER_UNIT = 30.0
WAGES = {'low': 10_320, 'mid': 18_635, 'high': 28_599}   # 최저임금 2026 / 비정규직 평균 / 정규직 평균 (2025.6)
MONEY = 10   # 0.1원 단위


def season_of(month: int) -> str:
    return 'summer' if month in (6, 7, 8) else ('winter' if month in (11, 12, 1, 2) else 'springfall')


def ratchet_K(month: int) -> int:
    return 12 if month in (7, 8, 9, 12, 1, 2) else 1


SAT_DR_PRICE = 40.0   # 봄·가을 토요일 11~14시: 한전 50% 할인 + 플러스DR 정산 → kWh당 31~50원 (정부 발표), 중간값


def tariff_korea(month: int, saturday_dr: bool = False) -> np.ndarray:
    light, mid, peak = PRICES[season_of(month)]
    if saturday_dr:   # 토요일: 최대부하 없이 중간부하, 11~14시는 할인 + 플러스DR
        e = np.array([light if (t * I.SLOT_H >= 22 or t * I.SLOT_H < 8) else mid for t in range(I.SLOTS)], dtype=float)
        e[I.slot_of(11):I.slot_of(14)] = SAT_DR_PRICE
        return e
    e = np.empty(I.SLOTS)
    for t in range(I.SLOTS):
        h = t * I.SLOT_H
        if h >= 22 or h < 8:
            e[t] = light
        elif month in (11, 12, 1, 2):
            e[t] = peak if (9 <= h < 12 or 16 <= h < 19) else mid
        else:
            e[t] = peak if 15 <= h < 21 else mid
    return e


def labor_korea(wage: int, crew: int, saturday: bool = False) -> Labor:
    regular = np.zeros(I.SLOTS, dtype=bool)
    blocks = []
    spans = [(6, 8), (8, 12), (12, 17), (17, 19), (19, 21), (21, 24)] if saturday else [(6, 8), (17, 19), (19, 21), (21, 24)]
    if not saturday:
        regular[I.slot_of(8):I.slot_of(17)] = True
    for a, b in spans:
        night_h = sum(1 for h in range(a, b) if h >= 22 or h < 6)
        cost = crew * wage * ((b - a) * 1.5 + night_h * 0.5)
        blocks.append((I.slot_of(a), I.slot_of(b), int(round(cost * MONEY))))
    return Labor(regular=regular, blocks=blocks)


def scaled_machines(machines: dict) -> dict:
    m = copy.deepcopy(machines)
    for v in m.values():
        v['consumption']['on'] = v['consumption']['on'] * KW_PER_UNIT
        v['consumption']['idle'] = v['consumption']['idle'] * KW_PER_UNIT
    return m


def build_korea_instance(machines, jobs, ops, seqs, month: int, alg: str, wage: int, pbar_ratio: float = 0.8,
                         saturday_dr: bool = False):
    mk = scaled_machines(machines)
    inst = D.build_instance(mk, jobs, ops, seqs, np.zeros(I.SLOTS), 32, alg)
    inst.tariff = Tariff(energy=np.round(tariff_korea(month, saturday_dr) * MONEY).astype(int), demand_rate=DEMAND_RATE * MONEY,
                         K=ratchet_K(month), season=season_of(month))
    inst.labor = labor_korea(wage, crew=len(mk), saturday=saturday_dr)
    base = np.full(I.SLOTS, 10.0)
    base[I.slot_of(6):I.slot_of(22)] = 20.0
    inst.base_load = base
    max_load = sum(v['consumption']['on'] for v in mk.values()) + 20.0
    inst.P_bar = pbar_ratio * max_load
    inst.month_max = 0.0
    inst.unfinished_penalty = 10_000_000 * MONEY
    inst.single_on = True
    return inst, mk


def won(x: float) -> float:
    return x / MONEY
