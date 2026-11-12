"""외부 예보 + 과거 오차로 '범위'와 '가능한 내일'을 만든다 (모델 학습 없음).

1. 과거 날마다 오차 = (실제 − 예보) ÷ 맑은 날 발전량 을 저장
2. 내일 예보가 주어지면, 계절이 비슷하고(앞뒤 1달) 예보 맑음 정도가 비슷한 과거 날 k개를 고름
3. 가운데 값 = 예보 + 그 날들 오차의 중앙값 (예보의 치우침 보정)
4. 범위(10~90%) = 그 날들 오차의 10%·90% 값 ± 여유 폭
   - 여유 폭은 날씨 종류(예보 기준 맑음·오락가락·흐림)마다 따로, 최근 날들에서 적중률 80%가 되도록 정함
   - 배율이 아니라 '더하는' 폭이라, 맑은 날처럼 범위가 거의 0인 경우도 바로잡힘
5. 가능한 내일 = 예보 + 그 날들 중 하나의 하루치 오차 + 여유 폭 안의 하루 단위 흔들림
"""
from __future__ import annotations

import numpy as np


def _ratio(fc, clear):
    return float(fc.sum() / max(clear.sum(), 1e-9))


def _cls(r):
    return 0 if r < 0.45 else (1 if r < 0.8 else 2)      # 예보 기준 흐림 · 오락가락 · 맑음


class PVEstimator:
    def __init__(self, past_days: dict, k: int = 40, season_window: int = 1, calibrate: bool = True,
                 recent_days: int | None = None, target: float = 0.8, calib_month: int | None = None):
        """calib_month를 주면, 여유 폭을 그 달 앞뒤 1달에 해당하는 과거 날(모든 연도)로 맞춤 → 계절마다 반복되는 예보 성질 반영."""
        self.k, self.win, self.target = k, season_window, target
        self.dates = list(past_days.keys())
        self.index = {dt: i for i, dt in enumerate(self.dates)}
        self.keys = np.array([_ratio(d['fc'], d['clear']) for d in past_days.values()])
        self.months = np.array([dt.month for dt in self.dates])
        self.R = np.array([np.where(d['clear'] > 1e-6, (d['act'] - d['fc']) / np.maximum(d['clear'], 1e-6), 0.0)
                           for d in past_days.values()])
        self.delta = np.zeros(3)
        if calibrate:
            last = max(self.dates)
            if calib_month is not None:
                def near_month(m):
                    dm = abs(m - calib_month)
                    return min(dm, 12 - dm) <= season_window
                cal = {d: v for d, v in past_days.items() if near_month(d.month)}
            else:
                cal = {d: v for d, v in past_days.items() if recent_days is None or (last - d).days < recent_days}
            self.delta = self._calibrate(cal)

    # 하위 호환: 예전 코드가 scale을 출력함
    @property
    def scale(self):
        return float(np.mean(self.delta))

    def _near(self, fc, clear, month=None, exclude=None):
        dist = np.abs(self.keys - _ratio(fc, clear))
        if month is not None:
            dm = np.abs(self.months - month)
            dm = np.minimum(dm, 12 - dm)
            dist = dist + np.where(dm <= self.win, 0.0, 10.0)
        if exclude is not None:
            dist[exclude] = np.inf
        return self.R[np.argsort(dist)[:self.k]]

    def _raw(self, fc, clear, month=None, exclude=None):
        R = self._near(fc, clear, month, exclude)
        med = np.median(R, axis=0)
        return np.quantile(R, 0.1, axis=0), med, np.quantile(R, 0.9, axis=0)

    def predict(self, fc, clear, month=None, scale=None, exclude=None):
        lo, med, hi = self._raw(fc, clear, month, exclude)
        d = self.delta[_cls(_ratio(fc, clear))]
        top = clear * 1.3
        q50 = np.clip(fc + med * clear, 0, top)
        q10 = np.clip(fc + (lo - d) * clear, 0, top)
        q90 = np.clip(fc + (hi + d) * clear, 0, top)
        return np.minimum(q10, q50), q50, np.maximum(q90, q50)

    def scenarios(self, fc, clear, n, rng, month=None):
        R = self._near(fc, clear, month)
        d = self.delta[_cls(_ratio(fc, clear))]
        pick = R[rng.integers(0, len(R), n)]
        shift = rng.uniform(-1, 1, (n, 1)) * max(d, 0.0)
        return np.clip(fc + (pick + shift) * clear, 0, clear * 1.3)

    def _calibrate(self, days):
        """같은 날을 빼고 범위를 만든 뒤, 실제가 범위 밖으로 벗어난 정도의 80% 값을 여유 폭으로 (날씨 종류별)."""
        E = {0: [], 1: [], 2: []}
        for dt, v in days.items():
            m = v['clear'] > 0.05 * v['clear'].max()
            lo, _, hi = self._raw(v['fc'], v['clear'], dt.month, self.index.get(dt))
            y = (v['act'] - v['fc']) / np.maximum(v['clear'], 1e-6)
            E[_cls(_ratio(v['fc'], v['clear']))].extend(np.maximum(lo - y, y - hi)[m])
        allE = np.concatenate([np.array(e) for e in E.values() if len(e)]) if any(len(e) for e in E.values()) else np.zeros(1)
        fallback = float(np.quantile(allE, self.target))
        return np.array([float(np.quantile(E[c], self.target)) if len(E[c]) >= 200 else fallback for c in range(3)])
