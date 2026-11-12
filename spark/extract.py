"""SPARK 기계 zip을 풀지 않고 바로 읽어서, 유효전력을 1분 · 15분 평균(kW)으로 줄인 작은 파일을 만든다.

zip 안 구조: <기계>/<측정항목>/<연도>_<측정항목>.csv.xz  (열: WsDateTime, 값)  5초 간격, 단위 W
- 시각은 세계 표준시(UTC)로 확인됨 (태양광 정오 출력 시각으로 확인) → 독일 현지 시각 열도 함께 저장
- P_total(3상 합계)이 없으면 P1+P2+P3로 계산

실행 예:
    py -m spark.extract --zips "C:\\spark\\10.35097-bjdg3m3rg5jv3skk\\data\\dataset\\TEC_*.zip" --year 2024
결과: data/spark_power/<기계>_<연도>_15min.csv, <기계>_<연도>_1min.csv.gz, summary.csv
"""
from __future__ import annotations

import argparse
import glob
import io
import lzma
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def _read_member(z: zipfile.ZipFile, name: str) -> pd.Series:
    with z.open(name) as f:
        raw = lzma.decompress(f.read()) if name.endswith('.xz') else f.read()
    df = pd.read_csv(io.BytesIO(raw))
    t = pd.to_datetime(df.iloc[:, 0])
    return pd.Series(df.iloc[:, 1].astype(float).values, index=t).sort_index()


def read_power(zip_path: str, year: int) -> tuple[str, pd.Series, str]:
    z = zipfile.ZipFile(zip_path)
    names = z.namelist()
    machine = Path(zip_path).stem

    def find(meas):
        hits = [n for n in names if n.endswith(f'/{meas}/{year}_{meas}.csv.xz') or n.endswith(f'/{meas}/{year}_{meas}.csv')]
        return hits[0] if hits else None

    for meas in ['P_total', 'AC_ActivePower']:
        m = find(meas)
        if m:
            return machine, _read_member(z, m), meas
    parts = [find(p) for p in ['P1', 'P2', 'P3']]
    if all(parts):
        s = sum(_read_member(z, p) for p in parts)
        return machine, s, 'P1+P2+P3'
    raise FileNotFoundError(f'{machine}: {year}년 유효전력 파일을 찾지 못함')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zips', required=True, help='기계 zip 경로 패턴 (여러 개면 쉼표로 구분)')
    ap.add_argument('--year', type=int, default=2024)
    ap.add_argument('--out', default='data/spark_power')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    files = sorted(set(f for pat in a.zips.split(',') for f in glob.glob(pat.strip())))
    files = [f for f in files if not Path(f).stem.startswith(('00', '01', '02'))]
    if not files:
        raise SystemExit('zip 파일을 찾지 못함: ' + a.zips)
    rows = []
    for f in files:
        print('읽는 중:', Path(f).name, flush=True)
        try:
            machine, s, meas = read_power(f, a.year)
        except Exception as e:
            print('  건너뜀:', e); continue
        kw = s / 1000.0
        m1 = kw.resample('1min').mean()
        cnt = kw.resample('15min').count()
        m15 = kw.resample('15min').mean().where(cnt >= 90)          # 15분(180개) 중 절반 이상 있어야 인정
        local = lambda idx: idx.tz_localize('UTC').tz_convert('Europe/Berlin').tz_localize(None)
        pd.DataFrame({'utc': m15.index, 'local': local(m15.index), 'kw': m15.round(3).values}).to_csv(out / f'{machine}_{a.year}_15min.csv', index=False)
        pd.DataFrame({'utc': m1.index, 'local': local(m1.index), 'kw': m1.round(3).values}).to_csv(out / f'{machine}_{a.year}_1min.csv.gz', index=False, compression='gzip')
        v = m15.dropna()
        rows.append(dict(machine=machine, measurement=meas, valid_15min=round(m15.notna().mean(), 3),
                         mean_kw=round(v.mean(), 2), p05_kw=round(v.quantile(.05), 2), p50_kw=round(v.median(), 2),
                         p95_kw=round(v.quantile(.95), 2), max_kw=round(v.max(), 2)))
        print(f'  {meas} · 15분 유효 {rows[-1]["valid_15min"]:.0%} · 평균 {rows[-1]["mean_kw"]} kW · 95% {rows[-1]["p95_kw"]} kW · 최대 {rows[-1]["max_kw"]} kW', flush=True)
    pd.DataFrame(rows).to_csv(out / 'summary.csv', index=False)
    print('\n저장:', out)


if __name__ == '__main__':
    main()
