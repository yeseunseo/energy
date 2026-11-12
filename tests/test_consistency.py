"""① 계산 검증 (2): 하루 전 계획의 '계획상 비용'과, 그 계획을 시뮬레이터로 실제 돌린 비용이 같은지 확인."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from peakshift.instance import generate_instance
from peakshift.forecast import PVForecaster
from peakshift import sim as S, planner
from peakshift.policies import EDDPolicy, PlanPolicy

for seed in (1, 2, 3):
    inst = generate_instance(seed, 16, 'summer', 'tight')
    q50 = PVForecaster(inst.pv_cap, 'summer', seed=seed).predict(0.25)[1]
    st0 = S.State.initial(inst, q50, [], keep_history=False); S.run(st0, inst, EDDPolicy())
    plan = planner.solve(inst, q50, time_limit=20, hint=planner.hint_from_run(st0))
    # 규칙 수리(피크 연기·끄기)를 끈 순수 계획 추종
    pol = PlanPolicy(plan, slack_guard=10**9)
    st = S.State.initial(inst, q50, [])
    pol.install(st); S.run(st, inst, pol)
    c = S.cost_breakdown(st, inst)
    diff = (c['total'] - plan.objective) / plan.objective * 100
    print(f'seed {seed}: 계획상 {plan.objective:>10,.0f}원 · 시뮬레이션 {c["total"]:>10,.0f}원 · 차이 {diff:+.2f}% · 미완료 {c["unfinished"]}')
