"""형세 판단(가치 함수): 지금 상태에서 '남은 하루 비용'을 바로 예측하는 학습 모델.

알파고의 '수읽기 + 형세 판단'과 같은 구조
  - 수읽기: 롤아웃이 후보 행동마다 몇 시간만 미래를 굴려 봄
  - 형세 판단: 그 뒤의 남은 비용은 이 모델이 바로 예측 → 하루 끝까지 굴리지 않아도 됨

예측 대상 = (하루 끝 비용) − (지금까지 든 비용), 하루 끝 비용은 롤아웃의 '하루 끝 비용 현실화(D2)'와 같은 정의.
"""
from __future__ import annotations

import numpy as np

from . import instance as I
from . import sim as S


def cost_so_far(st, inst) -> float:
    c = S.cost_breakdown(st, inst)
    return c['total'] - c['unfinished_pen']


def final_cost(st, inst, late_mult: float, job_cost: float) -> float:
    """롤아웃 D2와 같은 하루 끝 비용: 지금까지 비용 + 못 끝낸 공정을 평균 가격으로 처리하는 비용 + 지연 비용."""
    price = float(np.mean(inst.tariff.energy))
    rem_kwh, late = 0.0, 0
    for j in st.jobs:
        rem = [o for o in j.ops if st.op_status[o] != S.DONE]
        if rem:
            late += 1
            for o in rem:
                p, kw = list(st.ops[o].proc.values())[0]
                left = st.op_rem[o] if st.op_status[o] == S.RUN else p
                rem_kwh += kw * max(left, 0) * I.SLOT_H
    return cost_so_far(st, inst) + price * rem_kwh + late_mult * job_cost * late


def job_cost_of(inst) -> float:
    price = float(np.mean(inst.tariff.energy))
    per = [sum(kw * p for (p, kw) in (list(inst.ops[o].proc.values())[0] for o in j.ops)) for j in inst.jobs]
    return price * float(np.mean(per)) * I.SLOT_H


FEATURES = None


def features(st, inst, pv_q50: np.ndarray) -> np.ndarray:
    """상태 → 고정 길이 숫자 벡터."""
    t = st.t
    left = I.SLOTS - t
    price = inst.tariff.energy.astype(float)
    fut = price[t:] if t < I.SLOTS else np.array([0.0])
    pv = np.asarray(pv_q50, dtype=float)
    pvf = pv[t:] if t < I.SLOTS else np.array([0.0])
    M = len(inst.machines)
    work = np.zeros(M)      # 기계별 남은 처리 칸
    kwh = np.zeros(M)       # 기계별 남은 전력량
    for o in range(len(st.ops)):
        if st.op_status[o] == S.DONE:
            continue
        m, (p, kw) = list(st.ops[o].proc.items())[0]
        r = st.op_rem[o] if st.op_status[o] == S.RUN else p
        work[m] += r
        kwh[m] += r * kw * I.SLOT_H
    chains, unfinished = [], 0
    for j in st.jobs:
        rem = [o for o in j.ops if st.op_status[o] != S.DONE]
        if rem:
            unfinished += 1
            chains.append(sum(st.op_rem[o] if st.op_status[o] == S.RUN else list(st.ops[o].proc.values())[0][0] for o in rem))
    max_chain = max(chains) if chains else 0
    need = int(min(left, max(work.max(), 1)))
    cheap = np.sort(fut)[:need].mean() if len(fut) else 0.0      # 남은 일을 가장 싼 칸에 넣는다면의 평균 가격
    states = np.zeros((M, 4))
    for m in range(M):
        states[m, min(st.m_state[m], 3)] = 1
    f = [t, left, unfinished, max_chain, sum(chains), work.sum(), kwh.sum(),
         max_chain / max(left, 1), work.max() / max(left, 1),
         fut.mean(), fut[:8].mean(), fut[:16].mean(), fut.min(), fut.max(), cheap,
         pvf.sum() * I.SLOT_H, pvf[:8].sum() * I.SLOT_H, pvf[:16].sum() * I.SLOT_H,
         float(np.mean(inst.tariff.energy))]
    f += list(work) + list(kwh) + list(states.ravel())
    # 보강: 계획 대비 지연, 남은 전력량의 최소·평균 비용, 태양광 반영 순가격, 급주문 수
    behind = sum(max(0, t - st.pstart[o]) for o in st.pstart if st.op_status[o] == S.WAIT) if st.pstart else 0
    tot_kwh = kwh.sum()
    maxload = sum(m.p_proc for m in inst.machines)
    net = fut * np.clip(1 - pvf[:len(fut)] / max(maxload, 1e-6), 0, 1) if len(fut) else np.array([0.0])
    k = max(1, int(round(work.sum() / max(M, 1))))
    f += [behind, tot_kwh * cheap, tot_kwh * fut.mean(), tot_kwh * np.sort(net)[:k].mean(), net.mean(),
          sum(1 for j in st.jobs if j.urgent), max(chains) * (1.2) - left if chains else -left]
    return np.array(f, dtype=float)


class ValueModel:
    """LightGBM 회귀: features → 남은 비용."""

    def __init__(self, model):
        self.model = model

    def predict(self, st, inst, pv_q50) -> float:
        return float(self.model.predict(features(st, inst, pv_q50)[None, :])[0])
