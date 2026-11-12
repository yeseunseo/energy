"""실제 태양광 데이터로 하루치 예측 · 범위 · 가능한 내일 · 실제값을 제공한다.

- 예보: Open-Meteo 하루 전 일사량 예보 → 달별 계수로 발전량 환산 (pv.kosep)
- 범위 · 가능한 내일: 그 달 이전 날들의 오차로 만든 추정기 (pv.estimator)
- 실제값(채점용): 한국남동발전 구미태양광 실제 발전 기록을 공장 규모로 환산
- 계절은 한전 기준: 여름 6~8월, 봄·가을 3~5·9~10월, 겨울 11~2월. 평일만 사용(주말은 주간 계획 단계에서 다룸)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pv.kosep import build_days_real
from pv.estimator import PVEstimator
from . import instance as I

SEASON_MONTHS = {'summer': {6, 7, 8}, 'springfall': {3, 4, 5, 9, 10}, 'winter': {11, 12, 1, 2}}


class DayForecaster:
    """롤아웃(RolloutPolicy)이 기대하는 인터페이스: predict(_) → (q10, q50, q90), scenarios(_, n, rng) → (n, 96)."""

    def __init__(self, q, pool):
        self.q, self.pool = q, pool

    def predict(self, _cloud=None):
        return self.q

    def scenarios(self, _cloud, n, rng):
        return self.pool[rng.integers(0, len(self.pool), n)]


class RealPV:
    def __init__(self, openmeteo_csv: str, kosep_pattern: str, cap_kwp: float,
                 test_start: str = '2026-01-01', plant_kwp: float = 900.0, pool: int = 60, seed: int = 0):
        self.days, self.factors = build_days_real(openmeteo_csv, kosep_pattern, cap_kwp, train_end=test_start, plant_kwp=plant_kwp)
        self.test_start = pd.Timestamp(test_start)
        self.pool_n, self.seed = pool, seed
        self._est = {}

    def test_dates(self, season: str):
        months = SEASON_MONTHS[season]
        return sorted(d for d in self.days if d >= self.test_start and d.month in months and d.dayofweek < 5)

    def _estimator(self, d):
        key = (d.year, d.month)
        if key not in self._est:
            past = {x: v for x, v in self.days.items() if x < pd.Timestamp(d.year, d.month, 1)}
            self._est[key] = PVEstimator(past, k=40, calib_month=d.month)
        return self._est[key]

    def day(self, d):
        """그날의 (예보 범위 q10·q50·q90, 가능한 내일 묶음, 실제 발전량)."""
        v = self.days[d]
        est = self._estimator(d)
        q = est.predict(v['fc'], v['clear'], month=d.month)
        rng = np.random.default_rng(self.seed * 1_000_003 + int(d.strftime('%Y%m%d')))
        pool = est.scenarios(v['fc'], v['clear'], self.pool_n, rng, month=d.month)
        act = np.asarray(v['act'], dtype=float)
        rep = I.SLOTS // 96                       # 시간 단위가 15분보다 작으면 같은 값을 나눠 담음 (예: 5분 → 3칸)
        if rep > 1:
            q = tuple(np.repeat(x, rep) for x in q)
            pool = np.repeat(pool, rep, axis=1)
            act = np.repeat(act, rep)
        return DayForecaster(q, pool), act
