"""형세 판단 모델 학습: 시뮬레이터로 많은 하루를 돌려 (상태 → 남은 비용) 데이터를 만들고 LightGBM으로 학습.

- 무작위 하루 번호 100번 이후만 사용 (시험용 0~4, 연습용 5~9와 겹치지 않음)
- 태양광은 '가능한 내일' 묶음에서 뽑아 실제처럼 사용 (실제 기록 하나만 쓰면 다양성이 없음)
- 기준 계획을 따르되 결정마다 10% 확률로 일부러 미뤄, 롤아웃이 만나는 다양한 상태를 배움

실행 예:
    py train_value.py --jobs 5,10 --reps 40
결과: results_dunke/value_data.npz, results_dunke/value_model.txt, 학습 성능 출력
"""
from __future__ import annotations

import argparse
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from peakshift import dunke as D, sim as S, planner
from peakshift import instance as I
from peakshift.policies import PlanPolicy
from peakshift.pvreal import RealPV
from peakshift.value import features, cost_so_far, final_cost, job_cost_of
from run_dunke import KOR_DATE


def get_plan(inst, q50, key, cache_dir, time_limit):
    f = Path(cache_dir) / f'plan_{key}.pkl'
    if f.exists():
        return pickle.loads(f.read_bytes())
    pl = planner.solve(inst, q50, time_limit=time_limit)
    f.write_bytes(pickle.dumps(pl))
    return pl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', default='5,10')
    ap.add_argument('--reps', type=int, default=40)
    ap.add_argument('--rep-start', type=int, default=100)
    ap.add_argument('--eps', type=float, default=0.1)
    ap.add_argument('--late-mult', type=float, default=1.0)
    ap.add_argument('--spread', type=float, default=0.2)
    ap.add_argument('--urgent-rate', type=float, default=0.1)
    ap.add_argument('--plan-time', type=float, default=30)
    ap.add_argument('--pv-csv', default='data/pv_weather_hourly.csv')
    ap.add_argument('--kosep', default='data/kosep/*.csv')
    ap.add_argument('--machines', type=int, default=3, help='기계 수 (논문 3, 확장 실험 5)')
    ap.add_argument('--out', default='results_dunke')
    ap.add_argument('--train-only', action='store_true')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True)
    (out / 'plans').mkdir(exist_ok=True)
    data_f = out / f'value_data_{a.machines}m.npz'

    if not a.train_only:
        base = 'dunke_code'
        prices = D.load_prices(os.path.join(base, [f for f in os.listdir(base) if f.endswith('.xlsx')][0]))
        machines0 = D.generate_machines(a.machines)
        rpv = RealPV(a.pv_csv, a.kosep, cap_kwp=D.PV_CAP_RATIO * sum(m['consumption']['on'] for m in machines0.values()), seed=1)
        X, y, meta = [], [], []
        if data_f.exists():
            z = np.load(data_f, allow_pickle=True)
            X, y, meta = list(z['X']), list(z['y']), list(z['meta'])
        done = {(m[0], m[1], m[2]) for m in meta}
        t0 = time.time()
        for nj in [int(x) for x in a.jobs.split(',')]:
            alg = 'ex' if nj <= 5 and a.machines <= 3 else 'heu'
            for day in D.CALENDAR_DAYS:
                machines, jobs, ops, precs, seqs, seed = D.generate_instance_data(nj, day, a.machines)
                inst = D.build_instance(machines, jobs, ops, seqs, D.effective_price(prices[day], 'dyn', 'dyn'), 32, alg)
                fc, act = rpv.day(pd.Timestamp(KOR_DATE[day]))
                q50 = fc.predict()[1]
                plan = get_plan(inst, q50, f'{a.machines}m_{nj}_{day}', out / 'plans', a.plan_time)
                maker = D.urgent_maker(machines, nj)
                jc = job_cost_of(inst)
                for r in range(a.rep_start, a.rep_start + a.reps):
                    if (nj, day, r) in done:
                        continue
                    rng = np.random.default_rng(day * 1000 + r)
                    events = [(h * 4, 'urgent', maker(rng, h * 4)) for h in range(9, 17) if rng.random() < a.urgent_rate]
                    pact = D.sample_actual_times(inst.ops, rng, a.spread)
                    pv = fc.pool[rng.integers(len(fc.pool))]
                    st = S.State.initial(inst, pv, events, p_act=pact, keep_history=False)
                    pol = PlanPolicy(plan)
                    pol.install(st)
                    rows, csf = [], []
                    while st.t < I.SLOTS:
                        rows.append(features(st, inst, q50))
                        csf.append(cost_so_far(st, inst))
                        S.apply_events(st, inst)
                        acts = pol.decide(st, inst)
                        for m in list(acts):
                            if acts[m][0] == 'start' and rng.random() < a.eps:
                                acts[m] = ('wait',)
                        S.step(st, inst, acts)
                    F = final_cost(st, inst, a.late_mult, jc)
                    X += rows
                    y += [F - c for c in csf]
                    meta += [(nj, day, r)] * len(rows)
                print(f'{nj}개 작업 · {day}일 완료 ({time.time() - t0:.0f}초, 누적 {len(y)}행)', flush=True)
                np.savez_compressed(data_f, X=np.array(X), y=np.array(y), meta=np.array(meta))
    z = np.load(data_f, allow_pickle=True)
    X, y, meta = z['X'], z['y'], z['meta']
    import lightgbm as lgb
    reps = meta[:, 2].astype(int)
    cut = np.quantile(np.unique(reps), 0.8)
    tr, va = reps <= cut, reps > cut
    model = lgb.LGBMRegressor(n_estimators=1500, learning_rate=0.03, num_leaves=127, min_child_samples=30, verbose=-1)
    model.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], callbacks=[lgb.early_stopping(80, verbose=False)])
    pred = model.predict(X[va])
    mae = np.mean(np.abs(pred - y[va]))
    r2 = 1 - np.sum((pred - y[va]) ** 2) / np.sum((y[va] - y[va].mean()) ** 2)
    print(f'학습 {tr.sum()}행 · 검증 {va.sum()}행 · 검증 R² {r2:.3f} · 평균 절대 오차 {mae:,.0f} (남은 비용 평균 {np.mean(np.abs(y[va])):,.0f})')
    model.booster_.save_model(str(out / f'value_model_{a.machines}m.txt'))
    print('저장:', out / f'value_model_{a.machines}m.txt')


if __name__ == '__main__':
    main()
