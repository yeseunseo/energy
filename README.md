# PeakShift: 남는 태양광은 쓰고 비싼 피크는 피하는 중소 제조공장 실시간 생산 스케줄러

2026 동국대학교 공과대학 아이디어 경진대회(주제: 지속 가능한 에너지 솔루션) 출품작의 코드 저장소입니다.

PeakShift는 요금 시간대와 태양광 예보를 반영해 생산 일정을 자동으로 짜고, 급주문·작업 지연·예보 오차가 생기면 남은 하루를 미리 굴려 보고(롤아웃) 실시간으로 고칩니다.

## 구성

| 층 | 하는 일 | 코드 |
| --- | --- | --- |
| ① 태양광 발전량 추정 | 하루 전 일사량 예보 → 15분 발전량, 80% 범위, 가능한 내일 10개, 당일 보정 | `pv/` |
| ② 하루 전 계획 | 하루 15분 × 96칸 시간 색인 모형, CP-SAT | `peakshift/planner.py` |
| ③ 실시간 재조정 | 롤아웃(여유 보호, 하루 끝 비용) + 탄소 인지 + 결정 설명 | `peakshift/policies.py` |
| 연습용 공장 | 15분 단위 시뮬레이터, 비용 정산(전력량·래칫·인건비) | `peakshift/sim.py` |

## 폴더

```
peakshift/      핵심 패키지
  instance.py   공장·작업·요금·근무 블록 정의
  planner.py    CP-SAT 하루 전 계획 / 재계획
  sim.py        시뮬레이터, 비용 정산
  policies.py   A 기존 운영 · B 계획 따르기 · C 매번 다시 풀기 · D 롤아웃(+설명 기록)
  dunke.py      Dunke & Nickel(2025) 공장·작업 생성 이식
  korea.py      한국 2026 산업용(을) 요금 · 래칫 · 인건비 · 토요일 플러스DR
  pvreal.py     태양광 추정 결과를 실험에 연결
  value.py      학습 기반 형세 판단(시도, 본 방법에서는 제외)
pv/             태양광 발전량 추정 (예보 수집, 환산, 범위, 검증)
dunke_code/     Dunke & Nickel(2025) 공개 코드·가격 데이터 (원저작자 코드)
tests/          비용 계산·일관성 테스트
results/        최종 결과 요약
archive/        초기 프로토타입과 예비 실험 결과 (참고용)
spark/          KIT SPARK 데이터 추출 스크립트
```

## 설치

```
pip install -r requirements.txt
```

Windows에서는 `python` 대신 `py`를 써도 됩니다.

## 데이터 준비 (`data/` 폴더, 저장소에는 포함하지 않음)

1. **일사량 예보**: `python -m pv.fetch` → `data/pv_weather_hourly.csv` (Open-Meteo, 구미 2024.1~2026.10)
2. **구미 태양광 실적**: 공공데이터포털 [한국남동발전 시간대별 태양광 발전실적](https://www.data.go.kr/data/15131519/fileData.do) CSV를 `data/kosep/`에 저장
3. **Dunke & Nickel 가격 데이터**: `dunke_code/`에 포함

## 실행

```
# 테스트
python tests/test_costs.py

# ① 재현: 논문 실험 480개
python replicate_dunke.py
python replicate_tables.py

# ② 불확실성 실험 (연습용 하루는 --rep-offset 5, 시험용은 0)
python run_dunke.py --jobs 5 --reps 5 --rep-offset 0 --policies A,B,C,D12 --pv-csv data/pv_weather_hourly.csv --kosep "data/kosep/*.csv"
python run_dunke.py --jobs 10 --reps 5 --rep-offset 0 --policies A,B,C,D12 --pv-csv data/pv_weather_hourly.csv --kosep "data/kosep/*.csv"

# 탄소 인지 (탄소 가격 유로/톤)
python run_carbon.py --jobs 5 --reps 5 --rep-offset 0 --prices 0,800 --pv-csv data/pv_weather_hourly.csv --kosep "data/kosep/*.csv"

# ③ 한국 적용 (평일 / 봄·가을 토요일 + 플러스DR)
python run_korea.py --jobs 5 --reps 5 --rep-offset 0 --policies A,B,C,D12 --pv-csv data/pv_weather_hourly.csv --kosep "data/kosep/*.csv"
python run_korea.py --jobs 5 --reps 5 --rep-offset 0 --policies A,B,D12 --saturday-dr --pv-csv data/pv_weather_hourly.csv --kosep "data/kosep/*.csv"

# 결정 설명 시연
python explain_demo.py 258 10
```

## 주요 결과 (시험용 무작위 하루)

| 실험 | 결과 |
| --- | --- |
| ① 재현 | 논문 실험 480개, 평균 개선율 차이 1%p 이내 |
| ② 불확실성 | 완료율 96.0~97.1%(1위), 작업당 비용 논문 방식 대비 중앙값 2.8~5.5% 절감 (p 값 0.01 미만) |
| 탄소 | 논문 방식 대비 작업당 탄소 4.9~7.9%, 비용 2.3~4.5% 동시 감소 |
| ③ 한국 적용 | 작업당 총비용 기존 운영 대비 최대 15.8% 절감, 래칫 39% 감소 |

자세한 내용은 `results/결과_요약.md` 참고.

## 출처와 라이선스

- `dunke_code/`와 공장·작업 생성 방식: Dunke, F., & Nickel, S. (2025). Approximate and exact approaches to energy-aware job shop scheduling with dynamic energy tariffs and power purchase agreements. *Applied Energy*, 380, 125065. 원 저장소: gitlab.kit.edu/fabian.dunke/energyAwareJobShopScheduling (원저작자 라이선스를 따름)
- 배출 강도·기계 전력: KIT SPARK 데이터셋 (DOI 10.35097/bjdg3m3rg5jv3skk)
- 일사량 예보: Open-Meteo · 태양광 실적: 공공데이터포털 한국남동발전
- 요금: 한전 기본공급약관 별표1 (산업용(을), 2026)
