"""받은 SPARK 데이터 안에 어떤 기계 · 측정 항목 · 연도가 있는지 목록을 만든다.

SPARK 폴더 구조: 기계/측정항목/연도별 파일(.csv.xz)  (+ validation 폴더)
실행:
    py -m spark.inventory --root data\\spark
결과: spark_inventory.csv (전체 목록), 화면에 기계별 요약
"""
import argparse
import re
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='data/spark')
    ap.add_argument('--out', default='spark_inventory.csv')
    a = ap.parse_args()
    root = Path(a.root)
    if not root.exists():
        raise SystemExit(f'폴더가 없음: {root}')
    rows = []
    for f in root.rglob('*'):
        if not f.is_file():
            continue
        rel = f.relative_to(root).parts
        year = re.search(r'(20\d{2})', f.name)
        rows.append(dict(path=str(f.relative_to(root)), depth=len(rel), top=rel[0],
                         machine=rel[-3] if len(rel) >= 3 else '', measurement=rel[-2] if len(rel) >= 2 else '',
                         file=rel[-1], year=year.group(1) if year else '', mb=f.stat().st_size / 1e6))
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit('파일이 없음. 압축 파일이라면 먼저 풀어야 함')
    df.to_csv(a.out, index=False, encoding='utf-8-sig')
    print(f'파일 {len(df):,}개, 전체 {df.mb.sum() / 1000:.1f} GB')
    print('\n[가장 위 폴더]', sorted(df.top.unique())[:40])
    print('\n[확장자]', df.file.str.extract(r'(\.[^.]+(?:\.xz|\.gz)?)$')[0].value_counts().head(8).to_dict())
    data = df[df.file.str.contains(r'\.csv', regex=True)]
    g = data.groupby('machine').agg(측정항목수=('measurement', 'nunique'), 연도=('year', lambda y: ','.join(sorted(set(y) - {''}))), GB=('mb', lambda x: round(x.sum() / 1000, 2)))
    pd.set_option('display.width', 160, 'display.max_rows', 200)
    print('\n[기계별 요약]')
    print(g.to_string())
    pw = data[data.measurement.str.contains('power|Power|leistung|Leistung|P_', regex=True)]
    print('\n[이름에 power가 들어간 측정 항목 (기계별 앞 5개)]')
    for m, x in pw.groupby('machine'):
        print(f'  {m}: {sorted(x.measurement.unique())[:5]}')
    print(f'\n전체 목록 저장: {a.out}')


if __name__ == '__main__':
    main()
