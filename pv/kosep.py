"""한국남동발전 구미태양광 실제 발전 기록으로 '실제 발전량'을 만들고,
Open-Meteo 하루 전 일사량 예보를 '예보 발전량'으로 바꾼다.

왜: 실제 발전소 패널은 기울어져 있어 계절마다 일사량 대비 발전량이 크게 다르다 (맑은 날 기준 1월이 7월의 약 2.5배).
    수평면 일사량 × 고정 효율로는 이 모양이 안 나오므로, 과거 기록으로 달마다 '예보 일사량 → 실제 발전량' 관계를 맞춘다.

- 남동발전 파일: 줄 끝 쉼표 때문에 index_col=False로 읽어야 함. 'h시 발전량' = (h−1)~h시 1시간 발전량(kWh)
- Open-Meteo 일사량도 '직전 1시간 평균'이라 시각 기준이 같음
- 설비 용량: 공개 자료 미확인 → 데이터로 추정한 약 900kWp를 기본값으로 둠 (하루 평균 3,209kWh, 1시간 최대 810kWh와 부합)
"""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd

from pv.convert import clear_sky_ghi, to_15min, SLOTS

PLANT_KWP = 900.0


def load_kosep(pattern: str, plant: str = '구미태양광') -> pd.Series:
    """시간별 실제 발전량 (kW = 그 1시간 평균 출력). 인덱스는 구간 끝 시각 (Open-Meteo와 같은 기준)."""
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(pattern)
    df = pd.concat([pd.read_csv(f, encoding='utf-8-sig', index_col=False, skipinitialspace=True) for f in files], ignore_index=True)
    df.columns = [c.strip() for c in df.columns]
    df['발전구분'] = df['발전구분'].astype(str).str.strip()
    df = df[df['발전구분'] == plant].copy()
    df['일자'] = pd.to_datetime(df['일자'])
    df = df.drop_duplicates('일자').sort_values('일자')
    cols = [f'{h}시 발전량(KWh)' for h in range(1, 25)]
    vals = df[cols].astype(float).values.ravel()
    idx = (df['일자'].values[:, None] + (np.arange(1, 25) * np.timedelta64(1, 'h'))[None, :]).ravel()
    return pd.Series(vals, index=pd.DatetimeIndex(idx), name='kosep_kw')


def fit_monthly_factor(ghi_fc: pd.Series, act_kw: pd.Series, train_end: pd.Timestamp) -> dict:
    """달마다 실제 발전량 ≈ a_월 × 예보 일사량 (원점 통과 최소제곱, 학습 기간 낮 시간만)."""
    df = pd.DataFrame({'g': ghi_fc, 'p': act_kw}).dropna()
    df = df[(df.index < train_end) & (df.g > 20)]
    a = {}
    for m, x in df.groupby(df.index.month):
        a[m] = float((x.g * x.p).sum() / max((x.g ** 2).sum(), 1e-9))
    return a


def build_days_real(openmeteo_csv: str, kosep_pattern: str, cap_kwp: float, train_end: str,
                    plant_kwp: float = PLANT_KWP) -> tuple[dict, dict]:
    """하루 단위 배열 {날짜: dict(fc, act, clear)} (공장 규모 cap_kwp로 환산, 15분 kW)와 달별 계수."""
    om = pd.read_csv(openmeteo_csv, parse_dates=['time']).set_index('time').sort_index()
    act_h = load_kosep(kosep_pattern)
    a = fit_monthly_factor(om['ghi_fc1'], act_h, pd.Timestamp(train_end))
    scale = cap_kwp / plant_kwp
    fc_h = om['ghi_fc1'] * om.index.month.map(a).values * scale
    act_h = act_h.reindex(om.index) * scale
    q = pd.DataFrame({'fc': to_15min(fc_h), 'act': to_15min(act_h)})
    mid = q.index + pd.Timedelta(minutes=7.5)
    q['clear'] = clear_sky_ghi(pd.DatetimeIndex(mid)) * pd.Index(mid.month).map(a).values * scale
    days = {}
    for d, g in q.groupby(q.index.date):
        if len(g) != SLOTS or g[['fc', 'act']].isna().any().any():
            continue
        clear = np.maximum(g['clear'].values, 0)
        days[pd.Timestamp(d)] = dict(fc=np.clip(g['fc'].values, 0, clear * 1.3 + 1e-6), act=np.clip(g['act'].values, 0, None), clear=clear)
    return days, a
