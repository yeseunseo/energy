"""일사량 → 공장 태양광 발전량 (15분 단위, kW).

발전량 ≈ 설비 용량(kWp) × (일사량 ÷ 1000) × 효율 계수(기본 0.80)
맑은 날 이론 일사량은 태양 고도로 계산(Haurwitz 식)해서, '맑은 날 대비 비율'과 상한으로 쓴다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SLOTS = 96
LAT, LON = 36.1195, 128.3446


def cos_zenith(ts: pd.DatetimeIndex, lat=LAT, lon=LON) -> np.ndarray:
    """한국 표준시(UTC+9, 표준 경도 135°) 기준 태양 천정각의 코사인."""
    doy = ts.dayofyear.values
    hour = ts.hour.values + ts.minute.values / 60
    g = 2 * np.pi / 365 * (doy - 1 + (hour - 12) / 24)
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
            + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    eqt = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g) - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    solar_time = hour * 60 + eqt + 4 * (lon - 135.0)          # 분 단위
    ha = np.radians(solar_time / 4 - 180)
    phi = np.radians(lat)
    return np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(ha)


def clear_sky_ghi(ts: pd.DatetimeIndex) -> np.ndarray:
    cz = cos_zenith(ts)
    out = np.zeros_like(cz)
    m = cz > 0.01
    out[m] = 1098 * cz[m] * np.exp(-0.059 / cz[m])
    return out


def to_15min(hourly: pd.Series) -> pd.Series:
    """'직전 1시간 평균' 시간값을 그 시간 안의 15분 칸 4개에 나눠 담는다 (에너지 보존)."""
    s = hourly.copy()
    s.index = s.index - pd.Timedelta(hours=1)               # 값의 구간 시작으로 이동
    s = s.resample('15min').ffill(limit=3)
    return s


def build_days(hourly_csv: str, cap_kwp: float = 120.0, pr: float = 0.80) -> dict:
    """하루 단위 배열 사전: {날짜: dict(fc, act, clear)}  (모두 15분 kW, 길이 96)."""
    df = pd.read_csv(hourly_csv, parse_dates=['time']).set_index('time').sort_index()
    q = pd.DataFrame({c: to_15min(df[c]) for c in ['ghi_actual', 'ghi_fc1']})
    mid = q.index + pd.Timedelta(minutes=7.5)                 # 칸 가운데 시각으로 맑은 날 값 계산
    q['ghi_clear'] = clear_sky_ghi(pd.DatetimeIndex(mid))
    conv = cap_kwp / 1000 * pr
    days = {}
    for d, g in q.groupby(q.index.date):
        if len(g) != SLOTS or g[['ghi_actual', 'ghi_fc1']].isna().any().any():
            continue                                          # 하루가 온전하지 않으면 뺌
        clear = g['ghi_clear'].values * conv
        top = np.maximum(clear * 1.1, 1e-6)                    # 구름 가장자리 효과로 맑은 날을 약간 넘는 경우 허용
        days[pd.Timestamp(d)] = dict(
            fc=np.clip(g['ghi_fc1'].values * conv, 0, top),
            act=np.clip(g['ghi_actual'].values * conv, 0, top),
            clear=clear,
        )
    return days
