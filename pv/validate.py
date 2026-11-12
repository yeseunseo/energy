"""태양광 발전량 추정 검증.

시간 순서로 나눔: 오차 모음(과거) → 범위 보정 기간 → 시험 기간(가장 최근)
보는 것
  ① 가운데 값 오차 (낮 시간, 설비 용량 대비 %): 예보 그대로 / 치우침 보정 / 어제 값 / 같은 달 평균
  ② 10~90% 범위 적중률 (목표 80%)과 평균 폭
  ③ 가능한 내일 순위: 실제 하루 발전량이 시나리오 19개 중 몇 등인지 (고르게 퍼지면 정상)

실행:
    python -m pv.validate --csv data/pv_weather_hourly.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pv.convert import build_days
from pv.kosep import build_days_real
from pv.estimator import PVEstimator


def split(days, calib_start, test_start):
    a = {d: v for d, v in days.items() if d < calib_start}
    b = {d: v for d, v in days.items() if calib_start <= d < test_start}
    c = {d: v for d, v in days.items() if d >= test_start}
    return a, b, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='data/pv_weather_hourly.csv')
    ap.add_argument('--cap', type=float, default=120.0)
    ap.add_argument('--calib-start', default='2025-10-01')
    ap.add_argument('--test-start', default='2026-01-01')
    ap.add_argument('--out', default='results_pv')
    ap.add_argument('--kosep', default=None, help='남동발전 파일 경로 패턴 (예: "data/kosep/*.csv"). 주면 실제 발전 기록을 정답으로 씀')
    ap.add_argument('--plant-kwp', type=float, default=900.0, help='구미태양광 설비 용량 추정값')
    ap.add_argument('--fixed', action='store_true', help='예전 방식: 시험 기간 전체를 시험 시작 전 오차로만 맞춤')
    ap.add_argument('--recent', type=int, default=0, help='>0이면 범위 폭을 최근 며칠로 맞춤. 0(기본)이면 같은 계절(앞뒤 1달)의 과거 날로 맞춤')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True)

    factors = None
    if a.kosep:
        days, factors = build_days_real(a.csv, a.kosep, a.cap, train_end=a.test_start, plant_kwp=a.plant_kwp)
    else:
        days = build_days(a.csv, cap_kwp=a.cap)
    pool, calib, test = split(days, pd.Timestamp(a.calib_start), pd.Timestamp(a.test_start))
    print(f'온전한 날: 과거 {len(pool) + len(calib)}일 · 시험 {len(test)}일')
    past_all = {**pool, **calib}
    if a.fixed:
        est_for = lambda d: est_fixed
        est_fixed = PVEstimator(past_all, k=40, recent_days=None)
    else:
        # 실제 운영처럼: 매달 1일에 그 전까지 쌓인 모든 날로 다시 만들고, 범위 폭은 최근 a.recent일로 맞춤
        cache = {}
        def est_for(d):
            key = (d.year, d.month)
            if key not in cache:
                start = pd.Timestamp(d.year, d.month, 1)
                past = {x: v for x, v in days.items() if x < start}
                cache[key] = (PVEstimator(past, k=40, recent_days=a.recent) if a.recent > 0
                              else PVEstimator(past, k=40, calib_month=d.month))
            return cache[key]

    # 같은 달 평균 (기후값) 기준선
    clim = {}
    for d, v in pool.items():
        clim.setdefault(d.month, []).append(v['act'])
    clim = {m: np.mean(v, axis=0) for m, v in clim.items()}

    rng = np.random.default_rng(0)
    rows, ranks = [], []
    dates = sorted(test)
    for i, d in enumerate(dates):
        v = test[d]
        m = v['clear'] > 0.05 * v['clear'].max()
        est = est_for(d)
        q10, q50, q90 = est.predict(v['fc'], v['clear'], month=d.month)
        prev = days.get(d - pd.Timedelta(days=1))
        sc = est.scenarios(v['fc'], v['clear'], 19, rng, month=d.month)
        ranks.append(int(np.sum(sc.sum(axis=1) < v['act'].sum())))
        rows.append(dict(
            date=d, ratio=v['act'].sum() / max(v['clear'].sum(), 1e-9),
            mae_fc=np.mean(np.abs(v['act'][m] - v['fc'][m])),
            mae_q50=np.mean(np.abs(v['act'][m] - q50[m])),
            mae_prev=np.mean(np.abs(v['act'][m] - prev['act'][m])) if prev else np.nan,
            mae_clim=np.mean(np.abs(v['act'][m] - clim.get(d.month, np.zeros(96))[m])) if d.month in clim else np.nan,
            cover=np.mean((v['act'][m] >= q10[m]) & (v['act'][m] <= q90[m])),
            width=np.mean(q90[m] - q10[m]),
        ))
    r = pd.DataFrame(rows)
    r['날씨'] = pd.cut(r.ratio, [-1, 0.45, 0.8, 9], labels=['흐림', '오락가락', '맑음'])
    pct = lambda x: x / a.cap * 100

    lines = ['# 태양광 발전량 추정 검증 결과', '',
             f'- 설비 용량 {a.cap:.0f} kWp, 시험 기간 {dates[0].date()} ~ {dates[-1].date()} ({len(dates)}일)',
             ('- 정답: 남동발전 구미태양광 실제 발전 기록 (설비 용량 ' + f'{a.plant_kwp:.0f}kWp 추정, 공장 규모로 환산)' if a.kosep else '- 정답: Open-Meteo 실제 일사량으로 계산한 발전량'),
             ('- 달별 변환 계수 (예보 일사량 → 실제 발전량, 학습 기간에서 맞춤): ' + ', '.join(f'{m}월 {v:.2f}' for m, v in sorted(factors.items())) if factors else ''),
             f'- 하루 발전량 상관계수 (예보 vs 실제): {np.corrcoef([test[d]["fc"].sum() for d in dates], [test[d]["act"].sum() for d in dates])[0, 1]:.3f}',
             ('- 방식: 고정 (시험 시작 전 오차로만)' if a.fixed else ('- 방식: 매달 갱신 (그 달 이전 모든 날의 오차 사용, 범위 폭은 ' + (f'최근 {a.recent}일로 맞춤)' if a.recent > 0 else '같은 계절(앞뒤 1달) 과거 날로 맞춤)'))),
             '- 여유 폭 (흐림/오락가락/맑음, 맑은 날 발전량 대비 %): ' + ', '.join(f'{k[0]}-{k[1]:02d} ' + '/'.join(f'{x * 100:.1f}' for x in v.delta) for k, v in (cache.items() if not a.fixed else [((0, 0), est_fixed)])), '',
             '## ① 가운데 값 오차 (낮 시간, 설비 용량 대비 %)', '', '| 방법 | 오차 |', '| --- | --- |',
             f'| 예보 그대로 | {pct(r.mae_fc.mean()):.1f}% |',
             f'| 예보 + 치우침 보정 (제안) | {pct(r.mae_q50.mean()):.1f}% |',
             f'| 어제 실제값 그대로 | {pct(r.mae_prev.mean()):.1f}% |',
             f'| 같은 달 평균 | {pct(r.mae_clim.mean()):.1f}% |', '',
             '날씨별 (제안 방법)', '', '| 날씨 | 날 수 | 오차 | 범위 적중률 |', '| --- | --- | --- | --- |']
    for k, g in r.groupby('날씨', observed=True):
        lines.append(f'| {k} | {len(g)} | {pct(g.mae_q50.mean()):.1f}% | {g.cover.mean() * 100:.1f}% |')
    h = np.bincount(np.array(ranks) // 4, minlength=5)
    lines += ['', '## ② 10~90% 범위', '',
              f'- 적중률 {r.cover.mean() * 100:.1f}% (목표 80%), 평균 폭 {r.width.mean():.1f} kW', '',
              '## ③ 가능한 내일 순위 분포 (19개 중 몇 등, 5구간)', '',
              f'- {" · ".join(str(x) for x in h)}  (고르게 퍼지면 각 {len(dates) / 5:.0f} 근처)']
    (out / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    r.to_csv(out / 'daily_metrics.csv', index=False, encoding='utf-8-sig')
    print('\n'.join(lines))

    # 그림: 날씨별 예시 하루 3개
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, lab in zip(axes, ['맑음', '오락가락', '흐림']):
        g = r[r['날씨'] == lab]
        if g.empty:
            continue
        d = g.iloc[len(g) // 2].date
        v = test[d]
        q10, q50, q90 = est_for(d).predict(v['fc'], v['clear'], month=d.month)
        x = np.arange(96) / 4
        ax.fill_between(x, q10, q90, alpha=0.3, label='10~90% range')
        ax.plot(x, q50, lw=2, label='estimate (median)')
        ax.plot(x, v['fc'], ls='--', label='raw forecast')
        ax.plot(x, v['act'], color='k', lw=1.5, label='actual')
        ax.set_title(f'{d.date()}'); ax.set_xlim(4, 21); ax.set_xlabel('hour')
    axes[0].set_ylabel('kW'); axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / 'example_days.png', dpi=130)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(range(5), h); ax.axhline(len(dates) / 5, color='k', ls='--')
    ax.set_xticks(range(5)); ax.set_xticklabels(['0-3', '4-7', '8-11', '12-15', '16-19'])
    ax.set_xlabel('rank of actual among 19 scenarios'); ax.set_ylabel('days')
    fig.tight_layout(); fig.savefig(out / 'rank_histogram.png', dpi=130)
    print(f'\n저장: {out}/report.md, example_days.png, rank_histogram.png, daily_metrics.csv')


if __name__ == '__main__':
    main()
