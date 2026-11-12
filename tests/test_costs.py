"""① 계산 검증: 손으로 계산한 답과 시뮬레이터 결과가 같은지 확인한다.

기계 1대, 공정 1개짜리 작은 공장을 만들고, 정해진 시각에 정해진 행동을 하는 '각본 정책'으로 돌린다.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from peakshift.instance import (MachineSpec, OpSpec, JobSpec, Tariff, Instance, labor_example, SLOTS)
from peakshift import sim as S
from peakshift.policies import PlanPolicy

PRICE, RHO = 100, 8000
LAB = labor_example()


class Script:
    """각본 정책: {슬롯: {기계: 행동}}"""
    def __init__(self, plan):
        self.plan = plan

    def decide(self, st, inst):
        return dict(self.plan.get(st.t, {}))


def tiny(p=4, kw=10.0, due=60, P_bar=1000.0, K=12, base=0.0):
    m = MachineSpec('M', 'mill', p_idle=2.0, p_startup=5.0, warmup=1, p_proc=kw)
    op = OpSpec(0, 0, 0, 'mill', {0: (p, kw)})
    job = JobSpec(0, due, 1, (0,))
    return Instance(machines=[m], ops=[op], jobs=[job],
                    tariff=Tariff(np.full(SLOTS, PRICE), RHO, K, 'summer'), labor=LAB,
                    base_load=np.full(SLOTS, base), pv_cap=0, P_bar=P_bar, month_max=0.0,
                    lam_tardy=15_000, unfinished_penalty=1_000_000)


def run(inst, script, pv=None, events=()):
    pv = np.zeros(SLOTS) if pv is None else pv
    st = S.State.initial(inst, pv, list(events))
    S.run(st, inst, Script(script))
    return st, S.cost_breakdown(st, inst)


CASES = []


def case(fn):
    CASES.append(fn)
    return fn


@case
def energy_basic():
    """40슬롯에 켜면서 투입 → 40: 예열(대기 2 + 기동 5 = 7kW), 41~44: 가공 10kW, 45: 끄기."""
    st, c = run(tiny(), {40: {0: ('start', 0)}, 45: {0: ('off',)}})
    expect = PRICE * 0.25 * (7 + 4 * 10)
    assert abs(c['energy'] - expect) < 1e-6, (c['energy'], expect)
    assert st.op_end[0] == 45 and c['peak_kw'] == 10 and c['startups'] == 1
    return f"전력량 요금 {c['energy']:.0f}원 = {expect:.0f}원"


@case
def pv_offsets_load():
    """태양광 6kW → 슬롯별 한전 전력 = max(0, 부하 − 6). 꺼진 시간은 0 (음수로 내려가지 않음)."""
    _, c = run(tiny(), {40: {0: ('start', 0)}, 45: {0: ('off',)}}, pv=np.full(SLOTS, 6.0))
    expect = PRICE * 0.25 * ((7 - 6) + 4 * (10 - 6))
    assert abs(c['energy'] - expect) < 1e-6, (c['energy'], expect)
    return f"태양광 상계 {c['energy']:.0f}원 = {expect:.0f}원"


@case
def ratchet_summer():
    """기준치 5kW, 피크 10kW, 여름 K=12 → 8,000 × 12 × 5 = 480,000원."""
    _, c = run(tiny(P_bar=5.0), {40: {0: ('start', 0)}, 45: {0: ('off',)}})
    assert abs(c['ratchet'] - RHO * 12 * 5) < 1e-6, c['ratchet']
    return f"래칫(여름) {c['ratchet']:,.0f}원"


@case
def ratchet_springfall():
    """같은 조건, 봄·가을 K=1 → 40,000원."""
    _, c = run(tiny(P_bar=5.0, K=1), {40: {0: ('start', 0)}, 45: {0: ('off',)}})
    assert abs(c['ratchet'] - RHO * 1 * 5) < 1e-6, c['ratchet']
    return f"래칫(봄·가을) {c['ratchet']:,.0f}원"


@case
def ratchet_below_threshold():
    """피크가 기준치 아래면 래칫 0원."""
    _, c = run(tiny(P_bar=10.0), {40: {0: ('start', 0)}, 45: {0: ('off',)}})
    assert c['ratchet'] == 0
    return '기준치 이하 → 래칫 0원'


@case
def labor_block():
    """17:15(슬롯 69)에 투입 → 70~73 가공은 17~19시 블록 → 그 블록 비용 1회만."""
    _, c = run(tiny(due=96), {69: {0: ('start', 0)}, 74: {0: ('off',)}})
    expect = LAB.blocks[1][2]
    assert c['labor'] == expect, (c['labor'], expect)
    return f"연장 블록 비용 {c['labor']:,}원 (1회)"


@case
def labor_regular_free():
    """정규 시간 가공은 인건비 추가 없음."""
    _, c = run(tiny(), {40: {0: ('start', 0)}, 45: {0: ('off',)}})
    assert c['labor'] == 0
    return '정규 시간 → 추가 인건비 0원'


@case
def tardiness():
    """납기 43, 완료 45 → 2슬롯 × 15,000원 × 가중치 1 = 30,000원."""
    _, c = run(tiny(due=43), {40: {0: ('start', 0)}, 45: {0: ('off',)}})
    assert c['tardiness'] == 30_000, c['tardiness']
    return f"지연 {c['tardiness']:,.0f}원"


@case
def unfinished():
    """한 번도 투입 안 하면 미완료 벌점 1건."""
    _, c = run(tiny(), {})
    assert c['unfinished'] == 1 and c['unfinished_pen'] == 1_000_000
    return '미완료 벌점 1,000,000원'


@case
def idle_vs_off():
    """가공 후 5슬롯 켜 둠 → 대기 전력 2kW × 5슬롯이 더해진다."""
    _, c_off = run(tiny(), {40: {0: ('start', 0)}, 45: {0: ('off',)}})
    _, c_idle = run(tiny(), {40: {0: ('start', 0)}, 50: {0: ('off',)}})
    diff = c_idle['energy'] - c_off['energy']
    assert abs(diff - PRICE * 0.25 * 2 * 5) < 1e-6, diff
    return f"대기 5슬롯 추가 비용 {diff:.0f}원"


@case
def breakdown_resume():
    """가공 2슬롯째에 2슬롯 고장 → 수리 후 남은 2슬롯 재개(재예열 포함), 총 가공 4슬롯 유지."""
    inst = tiny()
    from peakshift.planner import Plan
    plan = Plan([[0]], {0: 41}, [np.ones(SLOTS, int)], [0, 0, 0, 0], 'X', 0, 0, 0, [])
    pol = PlanPolicy(plan)
    st = S.State.initial(inst, np.zeros(SLOTS), [(43, 'breakdown', (0, 2))])
    pol.install(st)
    S.run(st, inst, pol)
    assert st.op_status[0] == S.DONE, '고장 후 공정이 재개되지 않음'
    busy = sum(1 for g in st.load_hist if abs(g - 10) < 1e-9)
    assert busy == 4, busy
    return f'고장 후 재개 완료 (완료 슬롯 {st.op_end[0]}, 가공 4슬롯 유지)'


if __name__ == '__main__':
    ok = 0
    for fn in CASES:
        try:
            msg = fn()
            print(f'  통과  {fn.__name__:24s} {msg}')
            ok += 1
        except AssertionError as e:
            print(f'  실패  {fn.__name__:24s} {e}')
    print(f'\n{ok}/{len(CASES)} 통과')
