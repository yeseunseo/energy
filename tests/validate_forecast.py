"""⑥ 태양광 예측 검증 (가상 데이터 기준 — 방법 시연용).

1. 시간 순서 분할: 앞 365일로 학습, 그 뒤 새 200일로 검증
2. 가운데 값: 낮 시간 nMAE(설비 용량 대비 %) vs 기준선(어제 값, 같은 시각 과거 중앙값)
3. 범위: 10~90% 구간 적중률, 평균 폭, 분위수 손실과 기후값 대비 개선율
4. 날씨 입력 상·하한: 날씨 없음(하한) / 예보 구름량(제안) / 관측 구름량(상한, '완벽한 예보')
5. 시나리오: 실제 하루 발전량이 시나리오 19개 사이 몇 등인지 → 순위 분포가 고른지
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import lightgbm as lgb
from peakshift.forecast import clear_sky, cloud_path, pv_from_cloud
from peakshift.instance import SLOTS

CAP, SEASON = 120.0, 'summer'
cs = clear_sky(SEASON)
day = cs > 0.05 * cs.max()
Q = (0.1, 0.5, 0.9)


def make_days(n, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        c_day = rng.beta(1.2, 1.8)
        c_fc = float(np.clip(c_day + rng.normal(0, 0.15), 0, 1))
        out.append((c_day, c_fc, pv_from_cloud(rng, CAP, cs, cloud_path(rng, c_day))))
    return out


def feats(kind, c_day, c_fc):
    t = np.arange(SLOTS)
    if kind == 'none':
        return np.column_stack([t, cs])
    c = c_fc if kind == 'forecast' else c_day
    return np.column_stack([t, cs, np.full(SLOTS, c), cs * c])


def fit(train, kind):
    X = np.vstack([feats(kind, a, b) for a, b, _ in train]); Y = np.concatenate([p for _, _, p in train])
    ms = {}
    for q in Q:
        m = lgb.LGBMRegressor(objective='quantile', alpha=q, n_estimators=200, learning_rate=0.05, verbose=-1)
        ms[q] = m.fit(X, Y)
    return ms


def predict(ms, kind, c_day, c_fc):
    X = feats(kind, c_day, c_fc)
    p = np.array([np.clip(ms[q].predict(X), 0, CAP * cs) for q in Q])
    p.sort(axis=0)  # 분위수 순서 뒤집힘 정리
    return p


def pinball(y, qs):
    return np.mean([np.mean(np.maximum(q * (y - p), (q - 1) * (y - p))) for q, p in zip(Q, qs)])


train = make_days(365, 1)
test = make_days(200, 99)
clim = np.quantile(np.array([p for *_, p in train]), Q, axis=0)  # 같은 시각 과거 분포

res = {}
for kind in ('none', 'forecast', 'observed'):
    ms = fit(train, kind)
    mae, cov, wid, pin = [], [], [], []
    for c_day, c_fc, y in test:
        p = predict(ms, kind, c_day, c_fc)
        mae.append(np.mean(np.abs(y[day] - p[1][day])))
        cov.append(np.mean((y[day] >= p[0][day]) & (y[day] <= p[2][day])))
        wid.append(np.mean(p[2][day] - p[0][day]))
        pin.append(pinball(y[day], [pp[day] for pp in p]))
    res[kind] = dict(nmae=np.mean(mae) / CAP * 100, cov=np.mean(cov) * 100, wid=np.mean(wid), pin=np.mean(pin))
    if kind == 'forecast':
        model_fc = ms

# 기준선
mae_pers = np.mean([np.mean(np.abs(test[i][2][day] - test[i - 1][2][day])) for i in range(1, len(test))]) / CAP * 100
mae_clim = np.mean([np.mean(np.abs(y[day] - clim[1][day])) for *_, y in test]) / CAP * 100
pin_clim = np.mean([pinball(y[day], [c[day] for c in clim]) for *_, y in test])
cov_clim = np.mean([np.mean((y[day] >= clim[0][day]) & (y[day] <= clim[2][day])) for *_, y in test]) * 100

print('[가운데 값 정확도: 낮 시간 nMAE, 설비 용량 대비 %]')
print(f'  어제 값 그대로      {mae_pers:5.1f}%')
print(f'  같은 시각 과거 중앙값 {mae_clim:5.1f}%')
for k, name in [('none', '날씨 없음 (하한)'), ('forecast', '예보 구름량 (제안)'), ('observed', '관측 구름량 (상한)')]:
    print(f'  {name:16s}  {res[k]["nmae"]:5.1f}%')
print('\n[범위 예측: 10~90% 구간]')
print(f'  {"":16s}  적중률   평균 폭   분위수 손실   기후값 대비 개선')
print(f'  {"과거 분포(기후값)":14s}  {cov_clim:5.1f}%   {np.mean(clim[2][day] - clim[0][day]):5.1f}kW   {pin_clim:6.2f}        —')
for k, name in [('none', '날씨 없음 (하한)'), ('forecast', '예보 구름량 (제안)'), ('observed', '관측 구름량 (상한)')]:
    r = res[k]
    print(f'  {name:16s}  {r["cov"]:5.1f}%   {r["wid"]:5.1f}kW   {r["pin"]:6.2f}      {(1 - r["pin"] / pin_clim) * 100:+5.1f}%')

# 시나리오 순위 히스토그램 (날짜 단위 잔차 재표집)
val = make_days(73, 7)
resid = np.array([y - predict(model_fc, 'forecast', a, b)[1] for a, b, y in val])
rng = np.random.default_rng(0)
ranks = []
for c_day, c_fc, y in test:
    q50 = predict(model_fc, 'forecast', c_day, c_fc)[1]
    sc = np.clip(q50 + resid[rng.integers(0, len(resid), 19)], 0, CAP * cs)
    ranks.append(int(np.sum(sc.sum(axis=1) < y.sum())))
h = np.bincount(np.array(ranks) // 4, minlength=5)
print('\n[시나리오 순위 분포: 실제 하루 발전량이 시나리오 19개 중 몇 등인가 (5구간)]')
print('  ', ' · '.join(f'{c}' for c in h), f' (고르게면 각 {len(test) / 5:.0f} 근처)')

# ---- 보정 후 (peakshift.forecast.PVForecaster: 등각 보정 + 조건부 시나리오)
from peakshift.forecast import PVForecaster
F = PVForecaster(CAP, SEASON, seed=1)
cov2, wid2, rk2, mae2 = [], [], [], []
rng2 = np.random.default_rng(0)
for c_day, c_fc, y in test:
    lo, mid, hi = F.predict(c_fc)
    mae2.append(np.mean(np.abs(y[day] - mid[day])))
    cov2.append(np.mean((y[day] >= lo[day]) & (y[day] <= hi[day])))
    wid2.append(np.mean(hi[day] - lo[day]))
    rk2.append(int(np.sum(F.scenarios(c_fc, 19, rng2).sum(axis=1) < y.sum())))
print('\n[보정 후: 등각 보정 + 예보가 비슷한 날에서 오차 뽑기]')
print(f'  nMAE {np.mean(mae2) / CAP * 100:.1f}% · 10~90% 적중률 {np.mean(cov2) * 100:.1f}% · 평균 폭 {np.mean(wid2):.1f}kW')
print('  시나리오 순위 분포', ' · '.join(str(c) for c in np.bincount(np.array(rk2) // 4, minlength=5)), f' (고르게면 각 {len(test) / 5:.0f} 근처)')
