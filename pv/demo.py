"""오프라인 시험용 가짜 데이터. 실제 분석에는 쓰지 않는다.

Open-Meteo 결과와 같은 형식(time, ghi_actual, ghi_fc1)으로 만든다.
장마철(6월 말~7월)에 흐린 날이 많고, 예보는 실제보다 약간 낙관적이며 하루 안의 변동을 덜 잡는다고 가정.
"""
import numpy as np
import pandas as pd
from pv.convert import clear_sky_ghi


def make(start='2024-01-01', end='2026-10-31', seed=0, out='data/demo_weather_hourly.csv'):
    rng = np.random.default_rng(seed)
    t = pd.date_range(start, pd.Timestamp(end) + pd.Timedelta(hours=23), freq='h')
    clear = clear_sky_ghi(t - pd.Timedelta(minutes=30))
    act, fc = np.zeros(len(t)), np.zeros(len(t))
    for d in pd.date_range(start, end, freq='D'):
        idx = np.where(t.normalize() == d)[0]
        monsoon = 0.25 if (d.month == 7 or (d.month == 6 and d.day > 20)) else 0.0
        c_day = np.clip(rng.beta(1.3, 1.9) + monsoon, 0, 1)
        ar = np.zeros(24)
        for h in range(1, 24):
            ar[h] = 0.85 * ar[h - 1] + rng.normal(0, 0.09)
        c_act = np.clip(c_day + ar, 0, 1)
        c_fc = np.clip(c_day + rng.normal(0, 0.15) - 0.04, 0, 1)
        act[idx] = clear[idx] * (1 - 0.75 * c_act)
        fc[idx] = clear[idx] * (1 - 0.75 * c_fc)
    pd.DataFrame({'time': t.strftime('%Y-%m-%dT%H:%M'), 'ghi_actual': act.round(1), 'ghi_fc1': fc.round(1)}).to_csv(out, index=False)
    print('가짜 데이터 저장:', out)


if __name__ == '__main__':
    make()
