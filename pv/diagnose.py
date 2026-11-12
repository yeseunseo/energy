"""예보 성질이 시간에 따라 바뀌었는지 진단한다.

달마다 (낮 시간 기준)
  치우침 = 평균(실제 − 예보) ÷ 맑은 날 발전량   (+면 예보가 실제보다 낮음 = 비관적)
  흔들림 = 오차의 표준편차 ÷ 맑은 날 발전량
  맑은 날 비율

실행:
    py -m pv.diagnose --csv data/pv_weather_hourly.csv
"""
import argparse

import numpy as np
import pandas as pd

from pv.convert import build_days


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='data/pv_weather_hourly.csv')
    a = ap.parse_args()
    days = build_days(a.csv)
    rows = []
    for d, v in days.items():
        m = v['clear'] > 0.05 * v['clear'].max()
        r = (v['act'][m] - v['fc'][m]) / v['clear'][m]
        rows.append(dict(month=d.strftime('%Y-%m'), bias=r.mean(), spread=r.std(),
                         clear_day=v['act'].sum() / max(v['clear'].sum(), 1e-9) > 0.8))
    df = pd.DataFrame(rows).groupby('month').agg(날수=('bias', 'size'), 치우침=('bias', 'mean'), 흔들림=('spread', 'mean'), 맑은날비율=('clear_day', 'mean'))
    df[['치우침', '흔들림', '맑은날비율']] = (df[['치우침', '흔들림', '맑은날비율']] * 100).round(1)
    pd.set_option('display.width', 120)
    print('달별 예보 진단 (단위 %, 치우침이 +면 예보가 실제보다 낮음)')
    print(df.to_string())
    y = pd.DataFrame(rows)
    y['year'] = y.month.str[:4]
    print('\n연도별 평균 치우침 (%):', (y.groupby('year').bias.mean() * 100).round(1).to_dict())


if __name__ == '__main__':
    main()
