"""탄소 인지 실험: 결정 가격 = 전기 가격 + 탄소 가격 × 15분 배출 강도.

- 배출 강도: 2023년 독일 남서부 송전 구역(TransnetBW) 15분 값, 논문 가격과 같은 날짜·같은 시각 기준(UTC+1)
- 채점: 실제 전기 비용(탄소 가격 제외)과 실제 배출량(kg)을 따로 계산
- 탄소 가격을 바꿔 가며 '비용 vs 탄소' 곡선을 그림

실행 예:
    py run_carbon.py --jobs 5 --reps 2 --prices 0,50,100,200,400,800
"""
from __future__ import annotations

import argparse
import copy
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from peakshift import dunke as D, sim as S, planner
from peakshift import instance as I
from peakshift.policies import PlanPolicy, RolloutPolicy
from peakshift.pvreal import RealPV
from run_dunke import KOR_DATE


def load_ef(path='results_dunke/ef_2023_days.npy', exo=None):
    if os.path.exists(path):
        arr = np.load(path)
        return {d: arr[i] for i, d in enumerate(D.CALENDAR_DAYS)}
    e = pd.read_csv(exo)
    e['t'] = pd.to_datetime(e.utc_datetime, utc=True) + pd.Timedelta(hours=1)
    e = e.set_index('t')['emission_intensity_g_per_kwh']
    out = {}
    for day in D.CALENDAR_DAYS:
        d0 = pd.Timestamp('2023-01-01', tz='UTC') + pd.Timedelta(days=day - 1)
        out[day] = e[(e.index >= d0) & (e.index < d0 + pd.Timedelta(days=1))].values
    np.save(path, np.array([out[d] for d in D.CALENDAR_DAYS]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=int, default=5)
    ap.add_argument('--machines', type=int, default=3)
    ap.add_argument('--days', default='all')
    ap.add_argument('--reps', type=int, default=2)
    ap.add_argument('--rep-offset', type=int, default=5)
    ap.add_argument('--prices', default='0,50,100,200,400,800', help='탄소 가격 (유로/톤)')
    ap.add_argument('--policies', default='B,D12')
    ap.add_argument('--plan-time', type=float, default=20)
    ap.add_argument('--rollout-n', type=int, default=8)
    ap.add_argument('--spread', type=float, default=0.2)
    ap.add_argument('--urgent-rate', type=float, default=0.1)
    ap.add_argument('--exo', default='01exogenous_data/electricity_emissions_15min_transnetbw_2018_2024.csv')
    ap.add_argument('--pv-csv', default='data/pv_weather_hourly.csv')
    ap.add_argument('--kosep', default='data/kosep/*.csv')
    ap.add_argument('--out', default='results_dunke')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True)
    base = 'dunke_code'
    prices = D.load_prices(os.path.join(base, [f for f in os.listdir(base) if f.endswith('.xlsx')][0]))
    ef = load_ef(str(out / 'ef_2023_days.npy'), a.exo)
    machines0 = D.generate_machines(a.machines)
    rpv = RealPV(a.pv_csv, a.kosep, cap_kwp=D.PV_CAP_RATIO * sum(m['consumption']['on'] for m in machines0.values()), seed=1)
    days = D.CALENDAR_DAYS if a.days == 'all' else [int(x) for x in a.days.split(',')]
    alg = 'ex' if a.jobs <= 5 and a.machines <= 3 else 'heu'
    rows = []
    csv = out / f'carbon_{a.machines}m_{a.jobs}jobs.csv'
    for day in days:
        machines, jobs, ops, precs, seqs, seed = D.generate_instance_data(a.jobs, day, a.machines)
        price = D.effective_price(prices[day], 'dyn', 'dyn')
        fc, act = rpv.day(pd.Timestamp(KOR_DATE[day]))
        q10, q50, q90 = fc.predict()
        maker = D.urgent_maker(machines, a.jobs)
        for cp in [float(x) for x in a.prices.split(',')]:
            dec_price = price + cp * ef[day] / 1000.0            # €/MWh + (€/t × g/kWh ÷ 1000)
            inst = D.build_instance(machines, jobs, ops, seqs, dec_price, 32, alg)
            plan = planner.solve(inst, q50, time_limit=a.plan_time)
            for r in range(a.rep_offset, a.rep_offset + a.reps):
                rng = np.random.default_rng(day * 1000 + r)
                events = [(h * 4, 'urgent', maker(rng, h * 4)) for h in range(9, 17) if rng.random() < a.urgent_rate]
                pact = D.sample_actual_times(inst.ops, rng, a.spread)
                for p in a.policies.split(','):
                    if p == 'B':
                        pol = PlanPolicy(plan)
                    else:
                        pol = RolloutPolicy(plan, fc, None, maker, n_scen=a.rollout_n, k=2, seed=day + r, urgent_rate=a.urgent_rate,
                                            p_spread=a.spread, slack_guard=True, terminal='realistic')
                        pol.name = 'D12 롤아웃+둘다'
                    st = S.State.initial(inst, act, events, p_act=pact)
                    pol.install(st)
                    S.run(st, inst, pol)
                    g = np.array(st.g_hist)
                    energy = float(np.sum(price * g) * I.SLOT_H)          # €/MWh × kW × h → 논문 단위 (가격 × 사용량)
                    co2 = float(np.sum(ef[day] * g) * I.SLOT_H / 1000)    # g/kWh × kWh ÷ 1000 → kg (전력 단위 1 = 1 kW 가정)
                    done = sum(1 for j in st.jobs if st.op_status[j.ops[-1]] == S.DONE)
                    rows.append(dict(day=day, rep=r, carbon_price=cp, policy=pol.name, done=done, jobs_total=len(st.jobs),
                                     energy_cost=energy, co2_kg=co2))
                    print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(csv, index=False, encoding='utf-8-sig')
    df = pd.DataFrame(rows)
    g = df.groupby(['policy', 'carbon_price']).agg(완료=('done', 'sum'), 전체=('jobs_total', 'sum'), 비용=('energy_cost', 'sum'), 탄소=('co2_kg', 'sum'))
    g['작업당비용'] = g['비용'] / g['완료']
    g['작업당탄소kg'] = g['탄소'] / g['완료']
    g['완료율%'] = g['완료'] / g['전체'] * 100
    for pol in g.index.get_level_values(0).unique():
        b = g.loc[pol]
        b0 = b.loc[0.0] if 0.0 in b.index else b.iloc[0]
        g.loc[pol, '비용변화%'] = ((b['작업당비용'] / b0['작업당비용'] - 1) * 100).values
        g.loc[pol, '탄소변화%'] = ((b['작업당탄소kg'] / b0['작업당탄소kg'] - 1) * 100).values
    print('\n', g[['완료율%', '작업당비용', '작업당탄소kg', '비용변화%', '탄소변화%']].round(2).to_string())
    g.to_csv(out / f'carbon_summary_{a.machines}m_{a.jobs}jobs.csv', encoding='utf-8-sig')


if __name__ == '__main__':
    main()
