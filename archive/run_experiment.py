"""PeakShift 실험: 네 가지 운영 방식(A~D)을 돌발 상황별로 비교한다.

예시:
    python run_experiment.py                         # 빠른 기본 실행
    python run_experiment.py --season springfall --pbar loose --plan-time 60 --rollout-n 10
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from peakshift.instance import generate_instance, make_urgent_job, slot_of
from peakshift import instance as I
from peakshift.forecast import PVForecaster, realized_pv
from peakshift import sim as S
from peakshift import planner
from peakshift.policies import EDDPolicy, PlanPolicy, MPCPolicy, RolloutPolicy


def make_urgent_maker(inst):
    def maker(rng, slot):
        seed = int(rng.integers(1 << 30))

        def payload(jid, first_oid):
            return make_urgent_job(np.random.default_rng(seed), inst.machines, jid, first_oid, slot)
        return payload
    return maker


def make_events(kind, inst, seed):
    rng = np.random.default_rng(seed + 3000)
    maker = make_urgent_maker(inst)
    if kind == 'urgent':
        return [(slot_of(11), 'urgent', maker(rng, slot_of(11))),
                (slot_of(14), 'urgent', maker(rng, slot_of(14)))]
    if kind == 'breakdown':
        return [(slot_of(10.5), 'breakdown', (0, 8))]   # 밀링1, 10:30부터 2시간 고장
    return []


def sample_events(inst, seed, urgent_rate=0.15, breakdown_prob=0.2):
    """채점용 돌발 상황(급주문 · 고장)만 무작위로 뽑는다."""
    rng = np.random.default_rng(20_000 + seed)
    maker = make_urgent_maker(inst)
    events = []
    for h in range(9, 17):
        if rng.random() < urgent_rate:
            events.append((slot_of(h), 'urgent', maker(rng, slot_of(h))))
    if rng.random() < breakdown_prob:
        events.append((int(rng.integers(36, 64)), 'breakdown', (int(rng.integers(0, len(inst.machines))), int(rng.integers(4, 9)))))
    return events


def sample_day(inst, season, cloud_fc, seed, urgent_rate=0.15, breakdown_prob=0.2):
    """채점용 무작위 하루: 롤아웃이 쓰는 시나리오와 다른 난수로 뽑는다 (고르는 미래 ≠ 채점하는 미래)."""
    from peakshift.forecast import clear_sky, cloud_path, pv_from_cloud
    rng = np.random.default_rng(10_000 + seed)
    c_day = float(np.clip(cloud_fc + rng.normal(0, 0.15), 0, 1))
    pv = pv_from_cloud(rng, inst.pv_cap, clear_sky(season), cloud_path(rng, c_day))
    maker = make_urgent_maker(inst)
    events = []
    for h in range(9, 17):
        if rng.random() < urgent_rate:
            events.append((slot_of(h), 'urgent', maker(rng, slot_of(h))))
    if rng.random() < breakdown_prob:
        events.append((int(rng.integers(36, 64)), 'breakdown', (int(rng.integers(0, len(inst.machines))), int(rng.integers(4, 9)))))
    return pv, events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--jobs', type=int, default=16)
    ap.add_argument('--season', default='summer', choices=['summer', 'winter', 'springfall'])
    ap.add_argument('--pbar', default='tight', choices=['tight', 'loose'])
    ap.add_argument('--cloud', type=float, default=0.25, help='예보 구름량 0~1')
    ap.add_argument('--scenarios', nargs='+', default=['normal', 'urgent', 'pv_error', 'breakdown'])
    ap.add_argument('--policies', nargs='+', default=['A', 'B', 'C', 'D'])
    ap.add_argument('--plan-time', type=float, default=30.0)
    ap.add_argument('--mpc-time', type=float, default=5.0)
    ap.add_argument('--rollout-n', type=int, default=6)
    ap.add_argument('--rollout-k', type=int, default=3)
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--out', default='results')
    ap.add_argument('--days', type=int, default=0, help='>0이면 무작위 하루 R개로 기대 비용 평가')
    ap.add_argument('--urgent-rate', type=float, default=0.15, help='시간당 급주문 확률 (채점 하루와 롤아웃 사전분포 공통)')
    ap.add_argument('--pv', default='synthetic', choices=['synthetic', 'real'], help='real: 실제 하루 전 예보 + 구미 실제 발전 기록 (2026년 평일)')
    ap.add_argument('--pv-csv', default='data/pv_weather_hourly.csv')
    ap.add_argument('--kosep', default='data/kosep/*.csv')
    ap.add_argument('--pv-cap', type=float, default=None, help='공장 태양광 용량 kWp (기본 120)')
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(exist_ok=True)
    inst = generate_instance(a.seed, a.jobs, a.season, a.pbar)
    if a.pv_cap:
        inst.pv_cap = a.pv_cap
    print(f'[인스턴스] 작업 {len(inst.jobs)}개, 공정 {len(inst.ops)}개, 설비 {len(inst.machines)}대, '
          f'계절 {a.season}(K={inst.tariff.K}), 기준치 P̄={inst.P_ref:.0f} kW')

    t = time.time()
    rpv = None
    if a.pv == 'real':
        from peakshift.pvreal import RealPV
        rpv = RealPV(a.pv_csv, a.kosep, inst.pv_cap, seed=a.seed)
        cand = rpv.test_dates(a.season)
        pick = np.random.default_rng(a.seed).choice(len(cand), size=min(a.days, len(cand)), replace=False)
        real_dates = [cand[i] for i in sorted(pick)]
        fc, (q10, q50, q90) = None, (None, None, None)
        print(f'[태양광] 실제 데이터 · {a.season} 평일 {len(cand)}일 중 {len(real_dates)}일 채점 ({time.time() - t:.1f}초)')
    else:
        fc = PVForecaster(inst.pv_cap, a.season, seed=a.seed)
        q10, q50, q90 = fc.predict(a.cloud)
        print(f'[예측] LightGBM 분위수 학습 {time.time() - t:.1f}초, 예보 구름량 {a.cloud}')

    def make_plan(pv_q50, quiet=False):
        st0 = S.State.initial(inst, pv_q50, [], keep_history=False)
        S.run(st0, inst, EDDPolicy())
        pl = planner.solve(inst, pv_q50, time_limit=a.plan_time, workers=a.workers, hint=planner.hint_from_run(st0))
        if not quiet:
            gap = (pl.objective - pl.bound) / max(pl.objective, 1) * 100
            print(f'[하루 전 계획] {pl.status}, 목적값 {pl.objective:,.0f}원, 하한 {pl.bound:,.0f}원 '
                  f'(gap {gap:.1f}%), {pl.wall:.1f}초, 미배정 공정 {len(pl.skipped)}개, 근무 블록 {pl.z}')
        return pl

    plan = make_plan(q50) if rpv is None else None

    rows, traces = [], {}
    maker = make_urgent_maker(inst)
    if rpv is not None:
        days = []
        for i, d in enumerate(real_dates):
            dfc, act = rpv.day(d)
            days.append((d.strftime('%Y-%m-%d'), act, sample_events(inst, a.seed * 1000 + i, urgent_rate=a.urgent_rate), dfc))
    elif a.days > 0:
        days = [(f'day{d:02d}',) + sample_day(inst, a.season, a.cloud, a.seed * 1000 + d, urgent_rate=a.urgent_rate) for d in range(a.days)]
    else:
        days = [(scen, realized_pv(inst.pv_cap, a.season, a.cloud, scen, a.seed), make_events(scen, inst, a.seed))
                for scen in a.scenarios]
    for item in days:
        scen, pv, events = item[:3]
        if rpv is not None:
            fc = item[3]
            q10, q50, q90 = fc.predict()
            plan = make_plan(q50, quiet=True)
        for p in a.policies:
            if p == 'A':
                pol = EDDPolicy()
            elif p == 'B':
                pol = PlanPolicy(plan)
            elif p == 'C':
                pol = MPCPolicy(plan, (q10, q50, q90), time_limit=a.mpc_time, workers=a.workers)
            else:
                pol = RolloutPolicy(plan, fc, a.cloud, maker, n_scen=a.rollout_n, k=a.rollout_k, seed=a.seed, urgent_rate=a.urgent_rate)
            st = S.State.initial(inst, pv, events)
            if hasattr(pol, 'install'):
                pol.install(st)
            timer = []
            t = time.time()
            S.run(st, inst, pol, timer=timer)
            cb = S.cost_breakdown(st, inst)
            cb.update(scenario=scen, policy=pol.name, wall_s=round(time.time() - t, 1),
                      decide_ms_mean=1000 * float(np.mean(timer)), decide_ms_max=1000 * float(np.max(timer)),
                      pv_self_use_rate=cb['pv_self_use'] / max(pv.sum() * I.SLOT_H, 1e-9))
            rows.append(cb)
            traces[f'{scen}|{p}'] = dict(grid=st.g_hist, load=st.load_hist, pv=pv.tolist())
            print(f'  {scen:10s} {pol.name:18s} 총비용 {cb["total"]:>11,.0f}원  피크 {cb["peak_kw"]:6.1f}kW  '
                  f'미완료 {cb["unfinished"]}  ({cb["wall_s"]}초)')

    df = pd.DataFrame(rows)
    cols = ['scenario', 'policy', 'total', 'energy', 'ratchet', 'labor', 'tardiness', 'unfinished_pen',
            'peak_kw', 'startups', 'pv_self_use_rate', 'co2_kg', 'decide_ms_mean', 'decide_ms_max', 'wall_s']
    df[cols].to_csv(out / 'results.csv', index=False, encoding='utf-8-sig')
    (out / 'traces.json').write_text(json.dumps(traces))
    (out / 'tariff.json').write_text(json.dumps(dict(energy=inst.tariff.energy.tolist(), P_ref=inst.P_ref)))

    if a.days > 0 or rpv is not None:
        g = df.groupby('policy')
        summ = pd.DataFrame({
            '평균 총비용': g.total.mean(), '표준편차': g.total.std(), '평균 래칫': g.ratchet.mean(),
            '래칫 발생 비율': g.ratchet.apply(lambda x: (x > 0).mean()), '평균 인건비': g.labor.mean(),
            '평균 지연비용': g.tardiness.mean(), '평균 피크kW': g.peak_kw.mean(), '결정 평균 ms': g.decide_ms_mean.mean()})
        summ.to_csv(out / 'summary.csv', encoding='utf-8-sig')
        print('\n[무작위 하루 기대 비용]')
        print(summ.round(1).to_string())
        return
    piv = df.pivot(index='scenario', columns='policy', values='total').round(0)
    print('\n[총비용 (원)]')
    print(piv.to_string())
    base = df[df.policy.str.startswith('A')].set_index('scenario')['total']
    print('\n[A 대비 절감률 %]')
    print(((1 - piv.div(base, axis=0)) * 100).round(1).to_string())
    print(f'\n결과 저장: {out}/results.csv, {out}/traces.json')


if __name__ == '__main__':
    main()
