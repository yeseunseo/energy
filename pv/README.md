# 태양광 발전량 추정 (pv)

날씨를 직접 예측하지 않음. Open-Meteo의 "하루 전 일사량 예보"를 공장 태양광 발전량으로 바꾸고,
과거 예보 오차로 범위와 "가능한 내일"을 만든다.

## 실행 (인터넷 되는 컴퓨터에서)

    pip install requests pandas numpy matplotlib
    python -m pv.fetch --start 2024-01-01 --end 2026-10-31     # 구미 일사량 받기 (실제 + 하루 전 예보)
    python -m pv.validate --csv data/pv_weather_hourly.csv       # 검증 결과 → results_pv/

- 위치를 바꾸려면 fetch에 --lat --lon
- 태양광 용량을 바꾸려면 validate에 --cap (기본 120kWp)
- 시험 기간 시작은 --test-start (기본 2026-01-01). 그 전 날들이 "과거 오차"로 쓰임

## 파일

- fetch.py: Open-Meteo에서 받기 (archive = 실제, previous-runs의 _previous_day1 = 하루 전 예보)
- convert.py: 맑은 날 일사량 계산, 일사량 → 발전량 (용량 × 일사량/1000 × 0.80), 15분 칸으로 나누기
- estimator.py: 계절·예보가 비슷한 과거 날의 오차로 가운데 값 · 10~90% 범위 · 가능한 내일
- validate.py: 시간 순서 분할 검증 (오차, 범위 적중률, 시나리오 순위 분포, 그림)
- demo.py: 오프라인 시험용 가짜 데이터 (실제 분석에 쓰지 말 것)
