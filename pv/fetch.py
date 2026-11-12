"""Open-Meteo에서 구미의 시간별 일사량을 받는다.

- 실제 일사량: Historical Weather API (archive)
- 하루 전 예보: Previous Runs API, 변수 이름 끝에 _previous_day1 (2024년 1월부터 제공)
- 일사량(shortwave_radiation)은 "직전 1시간 평균" 값(W/m²)이다.

실행:
    python -m pv.fetch --start 2024-01-01 --end 2026-10-31
결과: data/pv_weather_hourly.csv  (time, ghi_actual, ghi_fc1)

비상업적 용도는 API 키 없이 무료. 한 달씩 나눠 받고, 요청 사이에 쉬어 간다.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests

LAT, LON = 36.1195, 128.3446          # 경북 구미시 (시청 부근)
TZ = 'Asia/Seoul'
ARCHIVE = 'https://archive-api.open-meteo.com/v1/archive'
PREVIOUS = 'https://previous-runs-api.open-meteo.com/v1/forecast'


def _get(url, params, tries=4):
    for k in range(tries):
        r = requests.get(url, params=params, timeout=60)
        if r.status_code == 200:
            return r.json()
        time.sleep(3 * (k + 1))
    r.raise_for_status()


def fetch_month(start: str, end: str, lat=LAT, lon=LON) -> pd.DataFrame:
    common = dict(latitude=lat, longitude=lon, start_date=start, end_date=end, timezone=TZ)
    a = _get(ARCHIVE, dict(common, hourly='shortwave_radiation'))
    f = _get(PREVIOUS, dict(common, hourly='shortwave_radiation_previous_day1'))
    da = pd.DataFrame({'time': a['hourly']['time'], 'ghi_actual': a['hourly']['shortwave_radiation']})
    df = pd.DataFrame({'time': f['hourly']['time'], 'ghi_fc1': f['hourly']['shortwave_radiation_previous_day1']})
    return da.merge(df, on='time', how='outer')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='2024-01-01')
    ap.add_argument('--end', default='2026-10-31')
    ap.add_argument('--lat', type=float, default=LAT)
    ap.add_argument('--lon', type=float, default=LON)
    ap.add_argument('--out', default='data/pv_weather_hourly.csv')
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    months = pd.date_range(a.start, a.end, freq='MS')
    parts = []
    for m in months:
        s = m.strftime('%Y-%m-%d')
        e = min(m + pd.offsets.MonthEnd(0), pd.Timestamp(a.end)).strftime('%Y-%m-%d')
        print('받는 중', s, '~', e)
        parts.append(fetch_month(s, e, a.lat, a.lon))
        time.sleep(1.0)
    df = pd.concat(parts).drop_duplicates('time').sort_values('time')
    df.to_csv(a.out, index=False)
    miss = df[['ghi_actual', 'ghi_fc1']].isna().mean() * 100
    print(f'저장: {a.out} ({len(df)}시간)  결측률 실제 {miss.ghi_actual:.1f}% · 예보 {miss.ghi_fc1:.1f}%')


if __name__ == '__main__':
    main()
