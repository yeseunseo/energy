"""② 불확실성 실험: Dunke & Nickel (2025) 공장 + 처리 시간 변동 · 급주문 · 태양광 예보 오차.

비교 방식
  A 기존 운영      : 논문의 순서 규칙(가장 많이 남은 일 우선)대로, 가격을 보지 않고 가장 이르게 처리
  B 논문 방식      : 하루 전 최적 계획(작업 5개: 정확, 10개: 순서 규칙 고정 = 논문과 같음)을 따르고 어긋나면 뒤로 밀기
  C 매번 다시 풀기 : 급주문·태양광 이탈이 생길 때마다 계획 전체를 다시 풂
  D PeakShift      : 하루 전 계획 + 롤아웃

실행 예:
    py run_dunke.py --jobs 5 --reps 3 --out results_dunke
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from peakshift import dunke as D, sim as S, planner
from peakshift.policies import EDDPolicy, PlanPolicy, MPCPolicy, RolloutPolicy
from peakshift.pvreal import RealPV

KOR_DATE = {15: '2026-01-15', 46: '2026-02-15', 74: '2026-03-15', 105: '2026-04-15', 135: '2026-05-15', 166: '2026-06-15',
            196: '2026-07-15', 227: '2026-08-15', 258: '2026-09-15', 288: '2026-10-15', 319: '2025-11-15', 349: '2025-12-15'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=int, default=5)
    ap.add_argument('--days', default='all', help='쉼표로 구분한 날짜 번호 또는 all')
    ap.add_argument('--reps', type=int, default=3, help='날짜마다 무작위 하루(불확실성 실현) 수')
    ap.add_argument('--start', type=int, default=32, help='작업 시작 가능 칸 (32 = 08:00)')
    ap.add_argument('--spread', type=float, default=0.2, help='처리 시간 변동 폭 ±')
    ap.add_argument('--urgent-rate', type=float, default=0.1, help='9~17시 시간당 급주문 확률')
    ap.add_argument('--plan-time', type=float, default=30)
    ap.add_argument('--mpc-time', type=float, default=5)
    ap.add_argument('--rollout-n', type=int, default=8)
    ap.add_argument('--policies', default='ABCD', help='A,B,C,D 외에 롤아웃 변형: D1(여유 보호) D2(하루 끝 비용) D12(둘 다). 쉼표로 구분 가능')
    ap.add_argument('--rep-offset', type=int, default=0, help='무작위 하루 번호 시작 (연습용은 5부터, 시험용은 0부터)')
    ap.add_argument('--late-mult', type=float, default=1.0)
    ap.add_argument('--value-model', default=None, help='DV: 형세 판단 롤아웃에 쓸 학습 모델')
    ap.add_argument('--horizon-h', type=float, default=2.0)
    ap.add_argument('--v-scen', type=int, default=16, help='DV가 굴려 볼 미래 수 (빨라진 만큼 늘림)')
    ap.add_argument('--v-k', type=int, default=3)
    ap.add_argument('--xlsx', default=None)
    ap.add_argument('--pv-csv', default='data/pv_weather_hourly.csv')
    ap.add_argument('--kosep', default='data/kosep/*.csv')
    ap.add_argument('--machines', type=int, default=3, help='기계 수 (논문 3, 확장 실험 5)')
    ap.add_argument('--out', default='results_dunke')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True)

    xlsx = a.xlsx
    if xlsx is None:
        base = 'dunke_code'
        xlsx = os.path.join(base, [f for f in os.listdir(base) if f.endswith('.xlsx')][0])
    prices = D.load_prices(xlsx)
    machines0 = D.generate_machines(a.machines)
    total_on = sum(m['consumption']['on'] for m in machines0.values())
    rpv = RealPV(a.pv_csv, a.kosep, cap_kwp=D.PV_CAP_RATIO * total_on, seed=1)
    days = D.CALENDAR_DAYS if a.days == 'all' else [int(x) for x in a.days.split(',')]
    alg = 'ex' if a.jobs <= 5 and a.machines <= 3 else 'heu'
    vm = None
    if a.value_model is None:
        a.value_model = f'results_dunke/value_model_{a.machines}m.txt'
    if any(p.startswith('DV') for p in (a.policies.split(',') if ',' in a.policies else [a.policies])) and os.path.exists(a.value_model):
        import lightgbm as lgb
        from peakshift.value import ValueModel
        vm = ValueModel(lgb.Booster(model_file=a.value_model))
    rows = []
    csv = out / f'results_{a.machines}m_{a.jobs}jobs_r{a.rep_offset}.csv'
    for day in days:
        machines, jobs, ops, precs, seqs, seed = D.generate_instance_data(a.jobs, day, a.machines)
        inst = D.build_instance(machines, jobs, ops, seqs, D.effective_price(prices[day], 'dyn', 'dyn'), a.start, alg)
        fc, act = rpv.day(pd.Timestamp(KOR_DATE[day]))
        q10, q50, q90 = fc.predict()
        t = time.time()
        plan = planner.solve(inst, q50, time_limit=a.plan_time)
        t_plan = time.time() - t
        base_plan = D.earliest_plan(inst, jobs, seqs, a.start)
        maker = D.urgent_maker(machines, a.jobs)
        for r in range(a.rep_offset, a.rep_offset + a.reps):
            rng = np.random.default_rng(day * 1000 + r)
            events = []
            for h in range(9, 17):
                if rng.random() < a.urgent_rate:
                    events.append((h * 4, 'urgent', maker(rng, h * 4)))
            pact = D.sample_actual_times(inst.ops, rng, a.spread)
            pols = a.policies.split(',') if ',' in a.policies else list(a.policies)
            for p in pols:
                if p == 'A':
                    pol = PlanPolicy(base_plan)
                    pol.name = 'A 기존 운영(가장 이르게)'
                elif p == 'B':
                    pol = PlanPolicy(plan)
                elif p == 'C':
                    pol = MPCPolicy(plan, (q10, q50, q90), time_limit=a.mpc_time)
                elif p.startswith('DV'):
                    pol = RolloutPolicy(plan, fc, None, maker, n_scen=a.v_scen, k=a.v_k, seed=day + r,
                                        urgent_rate=a.urgent_rate, p_spread=a.spread, slack_guard=True, terminal='realistic',
                                        late_mult=a.late_mult, value_model=vm, horizon_h=a.horizon_h)
                    pol.name = 'DV 롤아웃+형세판단'
                    if p == 'DV8':   # 같은 미래 수·후보 수로 속도만 비교
                        pol.n, pol.k = a.rollout_n, 2
                        pol.name = 'DV8 형세판단(미래8·후보2)'
                else:
                    pol = RolloutPolicy(plan, fc, None, maker, n_scen=a.rollout_n, k=2, seed=day + r,
                                        urgent_rate=a.urgent_rate, p_spread=a.spread,
                                        slack_guard=p in ('D1', 'D12'), terminal='realistic' if p in ('D2', 'D12') else None,
                                        late_mult=a.late_mult)
                    if p != 'D':
                        pol.name = {'D1': 'D1 롤아웃+여유보호', 'D2': 'D2 롤아웃+끝비용', 'D12': 'D12 롤아웃+둘다'}[p]
                st = S.State.initial(inst, act, events, p_act=pact)
                if hasattr(pol, 'install'):
                    pol.install(st)
                timer = []
                t = time.time()
                S.run(st, inst, pol, timer=timer)
                wall = time.time() - t + (t_plan if p != 'A' else 0)
                c = S.cost_breakdown(st, inst)
                done = sum(1 for j in st.jobs if st.op_status[j.ops[-1]] == S.DONE)
                rows.append(dict(day=day, rep=r, policy=pol.name, jobs_total=len(st.jobs), done=done,
                                 energy_cost=round(D.paper_cost(c['energy']), 3), grid_units=round(st.grid_kwh, 3),
                                 pv_used=round(st.pv_used, 3), urgent=len(events), sec=round(wall, 2), decide_sec=round(sum(timer), 3),
                                 replans=getattr(pol, 'replans', 0)))
                print(rows[-1], flush=True)
            pd.DataFrame(rows).to_csv(csv, index=False, encoding='utf-8-sig')
    df = pd.DataFrame(rows)
    g = df.groupby('policy').agg(완료작업=('done', 'mean'), 전체작업=('jobs_total', 'mean'), 에너지비용=('energy_cost', 'mean'),
                                 계산초=('sec', 'mean'), 결정초=('decide_sec', 'mean'), 재계획=('replans', 'mean'))
    print('\n', g.round(2).to_string())
    g['완료율%'] = g['완료작업'] / g['전체작업'] * 100
    g['작업당비용'] = g['에너지비용'] / g['완료작업']
    print('\n', g[['완료율%', '작업당비용']].round(1).to_string())
    g.to_csv(out / f'summary_{a.machines}m_{a.jobs}jobs_r{a.rep_offset}.csv', encoding='utf-8-sig')


if __name__ == '__main__':
    main()
