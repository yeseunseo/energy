"""태양광 발전량 예측: LightGBM 분위수 회귀 + 날짜 단위 잔차 부트스트랩 시나리오.

지금은 가상 이력 데이터(구름량 -> 발전량 생성 모형)로 학습한다.
실제 데이터를 쓸 때는 load_history()만 SPARK·기상청 데이터를 읽도록 바꾸면 된다.
"""
from __future__ import annotations

import numpy as np
import lightgbm as lgb

from . import instance as I

QUANTILES = (0.1, 0.5, 0.9)


def clear_sky(season: str) -> np.ndarray:
    rise, sset = {'summer': (5.5, 19.5), 'winter': (7.5, 17.5), 'springfall': (6.5, 18.5)}[season]
    h = (np.arange(I.SLOTS) + 0.5) * I.SLOT_H
    x = np.clip((h - rise) / (sset - rise), 0, 1)
    return 0.85 * np.sin(np.pi * x) ** 1.2


def cloud_path(rng, c_day: float, drift: np.ndarray | None = None) -> np.ndarray:
    """하루 구름량 경로. 하루 평균 c_day 주변을 AR(1)로 흔든다."""
    ar = np.zeros(I.SLOTS)
    for t in range(1, I.SLOTS):
        ar[t] = 0.92 * ar[t - 1] + rng.normal(0, 0.04)
    c = c_day + ar
    if drift is not None:
        c = c + drift
    return np.clip(c, 0, 1)


def pv_from_cloud(rng, cap, cs, cloud) -> np.ndarray:
    return np.clip(cap * cs * (1 - 0.75 * cloud) * (1 + rng.normal(0, 0.02, I.SLOTS)), 0, None)


def _features(cs, cloud_fc):
    t = np.arange(I.SLOTS)
    return np.column_stack([t, cs, np.full(I.SLOTS, cloud_fc), cs * cloud_fc])


class PVForecaster:
    """LightGBM 분위수 회귀 + 보정 + 조건부 시나리오.

    검증에서 드러난 두 문제를 고친 버전:
      ① 범위가 너무 좁음(적중률 66%) → 표본 밖 잔차로 등각(conformal) 보정해 10~90% 구간을 넓힘
      ② 시나리오가 낙관적·좁음 → 표본 밖 잔차를 '맑은 날 대비 비율'로 저장하고, 예보 구름량이 비슷한 날에서만 뽑음
    """

    def __init__(self, cap: float, season: str, n_days: int = 365, seed: int = 0, folds: int = 5, k_near: int = 60):
        self.cap, self.cs = cap, clear_sky(season)
        self.k_near = k_near
        rng = np.random.default_rng(seed + 1000)
        days = []
        for _ in range(n_days):
            c_day = rng.beta(1.2, 1.8)
            c_fc = float(np.clip(c_day + rng.normal(0, 0.15), 0, 1))
            days.append((c_fc, pv_from_cloud(rng, cap, self.cs, cloud_path(rng, c_day))))
        self.day_mask = self.cs > 0.05 * self.cs.max()
        # 시간 블록 교차적합: 각 블록을 빼고 학습해 그 블록의 표본 밖 잔차를 모은다
        self.pool, E = [], []
        for blk in np.array_split(np.arange(n_days), folds):
            out = set(blk.tolist())
            ms = self._fit([d for i, d in enumerate(days) if i not in out])
            for i in blk:
                c_fc, y = days[i]
                q10, q50, q90 = self._raw(ms, c_fc)
                self.pool.append((c_fc, np.where(self.day_mask, (y - q50) / (self.cap * self.cs + 1e-6), 0.0)))
                E.extend(np.maximum(q10 - y, y - q90)[self.day_mask])
        self.qhat = float(np.quantile(E, 0.8))
        self.models = self._fit(days)

    def _fit(self, days):
        X = np.vstack([_features(self.cs, c) for c, _ in days])
        Y = np.concatenate([y for _, y in days])
        ms = {}
        for q in QUANTILES:
            m = lgb.LGBMRegressor(objective='quantile', alpha=q, n_estimators=200,
                                  learning_rate=0.05, num_leaves=31, min_child_samples=20, verbose=-1)
            ms[q] = m.fit(X, Y)
        return ms

    def _raw(self, ms, cloud_fc):
        X = _features(self.cs, cloud_fc)
        p = np.array([np.clip(ms[q].predict(X), 0, self.cap * self.cs) for q in QUANTILES])
        p.sort(axis=0)
        return p

    def predict(self, cloud_fc: float):
        q10, q50, q90 = self._raw(self.models, cloud_fc)
        top = self.cap * self.cs
        return np.clip(q10 - self.qhat, 0, top), q50, np.clip(q90 + self.qhat, 0, top)

    def scenarios(self, cloud_fc: float, n: int, rng) -> np.ndarray:
        """예보 구름량이 비슷한 과거 날 k개의 하루치 오차를 통째로 뽑아 가운데 값에 더한다."""
        _, q50, _ = self.predict(cloud_fc)
        near = sorted(self.pool, key=lambda v: abs(v[0] - cloud_fc))[:self.k_near]
        idx = rng.integers(0, len(near), n)
        top = self.cap * self.cs
        return np.array([np.clip(q50 + near[i][1] * top, 0, top) for i in idx])


def realized_pv(cap, season, cloud_fc, kind, seed):
    """시험일의 실제 발전량. kind='pv_error'면 12시 이후 예보보다 훨씬 흐려진다."""
    rng = np.random.default_rng(seed + 2000)
    cs = clear_sky(season)
    c_day = float(np.clip(cloud_fc + rng.normal(0, 0.05), 0, 1))
    drift = None
    if kind == 'pv_error':
        drift = np.zeros(I.SLOTS)
        drift[48:] = 0.55
    return pv_from_cloud(rng, cap, cs, cloud_path(rng, c_day, drift))
