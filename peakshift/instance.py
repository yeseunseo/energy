"""공장 인스턴스 정의와 가상 인스턴스 생성기.

시간 단위는 15분 슬롯이며 하루는 96슬롯이다. 전력은 kW(15분 평균), 요금은 원 단위.
요금·임금 수치는 모두 '예시값'이다. 실제 실험 전에 한전 요금표와 실제 임금으로 교체할 것.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

SLOTS = 96
SLOT_H = 0.25  # 시간
SLOT_MIN = 15


def set_resolution(minutes: int):
    """시간 단위를 바꾼다 (기본 15분). 다른 모듈은 instance.SLOTS를 실행 중에 읽으므로 실험 시작 전에 한 번 호출할 것."""
    global SLOTS, SLOT_H, SLOT_MIN
    assert (24 * 60) % minutes == 0
    SLOT_MIN, SLOTS, SLOT_H = minutes, 24 * 60 // minutes, minutes / 60


def slot_of(hour: float) -> int:
    return int(round(hour / SLOT_H))


@dataclass(frozen=True)
class MachineSpec:
    name: str
    kind: str          # 'mill' | 'lathe' | 'press' | 'dry'
    p_idle: float      # 대기 전력 kW
    p_startup: float   # 기동 슬롯에 추가로 드는 전력 kW
    warmup: int        # 예열 슬롯 수 (예열 중에는 가공 불가)
    p_proc: float      # 가공 전력 기준값 kW (작업마다 ±20% 변동)


@dataclass(frozen=True)
class OpSpec:
    oid: int
    job: int
    seq: int
    kind: str
    proc: dict         # machine index -> (처리 슬롯 수, 가공 전력 kW)


@dataclass(frozen=True)
class JobSpec:
    jid: int
    due: int           # 납기 슬롯
    weight: int        # 지연 가중치
    ops: tuple         # 공정 oid 순서
    release: int = 0
    urgent: bool = False


@dataclass
class Tariff:
    energy: np.ndarray     # 슬롯별 전력량 요금 (원/kWh, 정수)
    demand_rate: int       # 기본요금 단가 (원/kW·월)
    K: int                 # 기준치 초과분이 유지되는 개월 수 (하계·동계 12, 봄·가을 1)
    season: str


@dataclass
class Labor:
    regular: np.ndarray            # 정규 근무 슬롯 여부
    blocks: list                   # [(시작 슬롯, 끝 슬롯, 비용 원)] 연장·야간 근무 블록
    block_of: np.ndarray = None    # 슬롯 -> 블록 번호 (-1: 없음)

    def __post_init__(self):
        self.block_of = np.full(SLOTS, -1, dtype=int)
        for k, (a, b, _) in enumerate(self.blocks):
            self.block_of[a:b] = k
        self.staffable = self.regular | (self.block_of >= 0)
        # staff_ok_from[t] = t부터 연속으로 가동 가능한 슬롯 수
        run = np.zeros(SLOTS + 1, dtype=int)
        for t in range(SLOTS - 1, -1, -1):
            run[t] = run[t + 1] + 1 if self.staffable[t] else 0
        self.staff_run = run

    def window_ok(self, start: int, p: int) -> bool:
        return 0 <= start and start + p <= SLOTS and self.staff_run[start] >= p


@dataclass
class Instance:
    machines: list
    ops: list
    jobs: list
    tariff: Tariff
    labor: Labor
    base_load: np.ndarray      # 조명·공조 등 기저부하 kW
    pv_cap: float              # 태양광 용량 kWp
    P_bar: float               # 현재 요금적용전력 (기본요금 기준치) kW
    month_max: float           # 이번 달 지금까지의 최대 부하 kW
    lam_tardy: int = 15_000    # 지연 1슬롯·가중치 1당 비용 (원)
    unfinished_penalty: int = 1_000_000  # 당일 미완료 작업 1건당 비용 (원)
    emission_factor: float = 0.45        # kgCO2/kWh (예시값)
    feed_in: float = 0.0                 # 남는 태양광을 팔 때 단가 (요금과 같은 단위, 한국 설정은 0)
    op_window: dict | None = None        # 공정 -> (가장 이른, 가장 늦은) 가공 시작 칸 (계획 모형 축소용, 선택)
    extra_prec: list | None = None       # (a, b): b는 a가 끝난 뒤 시작 (같은 작업끼리 순서 고정 등, 선택)
    battery: dict | None = None          # 배터리: cap, soc_min, soc_max, soc0 (kWh), p_max (kW), eta (효율), deg (요금 단위/kWh)
    single_on: bool = False              # 기계를 하루에 한 번만 켤 수 있음 (처음 켠 뒤 마지막 작업까지 대기 유지, Dunke & Nickel 2025)
    seed: int = 0

    @property
    def P_ref(self) -> float:
        return max(self.P_bar, self.month_max)


# ---------------------------------------------------------------- 설비·요금·근무

MACHINES = [
    MachineSpec('밀링1', 'mill', 3.0, 6.0, 1, 16.0),
    MachineSpec('밀링2', 'mill', 3.0, 6.0, 1, 16.0),
    MachineSpec('선반1', 'lathe', 2.0, 4.0, 1, 11.0),
    MachineSpec('선반2', 'lathe', 2.0, 4.0, 1, 11.0),
    MachineSpec('프레스', 'press', 4.0, 10.0, 2, 30.0),
    MachineSpec('건조로', 'dry', 6.0, 15.0, 4, 35.0),
]

# 공정 종류별 처리 슬롯 수 범위
PROC_RANGE = {'mill': (3, 7), 'lathe': (3, 7), 'press': (2, 4), 'dry': (3, 6)}


def tariff_example(season: str) -> Tariff:
    """2026년 개편 시간대 + '예시' 단가.

    시간대(3~10월 평일, 산업용(을)): 22~08시 경부하, 08~15시·21~22시 중간부하, 15~21시 최대부하
    (헤럴드경제 2026.5.13). 겨울철 시간대는 다르므로 winter는 근사값이다.
    단가는 가정값이다. 한전 요금표로 교체할 것.
    """
    prices = {'summer': (120, 165, 235), 'winter': (125, 165, 230), 'springfall': (120, 145, 180)}[season]
    light, mid, peak = prices
    e = np.empty(SLOTS, dtype=int)
    for t in range(SLOTS):
        h = t * SLOT_H
        if 15 <= h < 21:
            e[t] = peak
        elif h >= 22 or h < 8:
            e[t] = light
        else:
            e[t] = mid
    K = 1 if season == 'springfall' else 12
    return Tariff(energy=e, demand_rate=8_320, K=K, season=season)


def labor_example(wage: int = 15_000, crew: int = 3) -> Labor:
    """정규 08~17시 + 연장·야간 블록 4개. 연장 50%, 22~06시 야간 50% 추가 (근로기준법 가산율)."""
    regular = np.zeros(SLOTS, dtype=bool)
    regular[slot_of(8):slot_of(17)] = True
    blocks = []
    for a, b in [(6, 8), (17, 19), (19, 21), (21, 23)]:
        night_h = sum(1 for h in range(a, b) if h >= 22 or h < 6)
        cost = crew * wage * ((b - a) * 1.5 + night_h * 0.5)
        blocks.append((slot_of(a), slot_of(b), int(cost)))
    return Labor(regular=regular, blocks=blocks)


def base_load_example() -> np.ndarray:
    b = np.full(SLOTS, 15.0)
    b[slot_of(6):slot_of(22)] = 25.0
    b[slot_of(8):slot_of(17)] = 35.0
    return b


# ---------------------------------------------------------------- 작업 생성

def _make_ops(rng, jid, first_oid, machines):
    kinds = ['mill' if rng.random() < 0.5 else 'lathe']
    if rng.random() < 0.6:
        kinds.append('press')
    if rng.random() < 0.5:
        kinds.append('dry')
    ops = []
    for seq, kind in enumerate(kinds):
        lo, hi = PROC_RANGE[kind]
        p = int(rng.integers(lo, hi + 1))
        proc = {}
        for m, spec in enumerate(machines):
            if spec.kind == kind:
                proc[m] = (p, round(spec.p_proc * float(rng.uniform(0.8, 1.2)), 1))
        ops.append(OpSpec(first_oid + seq, jid, seq, kind, proc))
    return ops


def make_urgent_job(rng, machines, jid, first_oid, arrival):
    """급주문: 1~2공정, 도착 후 3~5시간 안에 납기."""
    ops = _make_ops(rng, jid, first_oid, machines)[:2]
    work = sum(min(p for p, _ in o.proc.values()) for o in ops)
    due = min(SLOTS, arrival + max(work + 4, int(rng.integers(12, 21))))
    job = JobSpec(jid, due, 3, tuple(o.oid for o in ops), release=arrival, urgent=True)
    return job, ops


def generate_instance(seed=0, n_jobs=18, season='summer', pbar='tight') -> Instance:
    rng = np.random.default_rng(seed)
    machines = list(MACHINES)
    ops, jobs = [], []
    for j in range(n_jobs):
        jops = _make_ops(rng, j, len(ops), machines)
        ops.extend(jops)
        work = sum(min(p for p, _ in o.proc.values()) for o in jops)
        earliest = slot_of(8) + work
        due = max(int(rng.integers(slot_of(12), slot_of(20))), earliest + 6)
        jobs.append(JobSpec(j, min(due, SLOTS), int(rng.integers(1, 4)), tuple(o.oid for o in jops)))
    P_bar = {'tight': 110.0, 'loose': 200.0}[pbar]
    return Instance(
        machines=machines, ops=ops, jobs=jobs,
        tariff=tariff_example(season), labor=labor_example(),
        base_load=base_load_example(), pv_cap=120.0,
        P_bar=P_bar, month_max=P_bar * 0.95, seed=seed,
    )
