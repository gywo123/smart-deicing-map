"""제설 대상 선정과 경로 생성을 위한 배차 계획 알고리즘.

현재 MVP의 핵심 파이프라인은 다음과 같다.

1. Knapsack 또는 Budgeted Maximum Coverage로 제설 대상 도로를 선정한다.
2. 선정된 대상을 단순 VRP nearest-neighbor 휴리스틱으로 경로화한다.

기존 Greedy 할당 함수는 가벼운 비교용 baseline으로 유지한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math


@dataclass(frozen=True)
class DeicingCandidate:
    """제설 처리 대상이 될 수 있는 도로 세그먼트 또는 grid cell 묶음."""

    candidate_id: str
    cost: int
    value: float
    covered_cells: frozenset[str] = field(default_factory=frozenset)
    x: float = 0.0
    y: float = 0.0
    service_time: float = 0.0
    demand: int = 1

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id must not be empty")
        if self.cost <= 0:
            raise ValueError("cost must be positive")
        if self.value < 0:
            raise ValueError("value must be non-negative")
        if self.service_time < 0:
            raise ValueError("service_time must be non-negative")
        if self.demand < 0:
            raise ValueError("demand must be non-negative")
        object.__setattr__(self, "covered_cells", frozenset(self.covered_cells))


@dataclass(frozen=True)
class RoutingVehicle:
    """VRP 방식 경로 생성에 사용할 차량 제약 조건."""

    vehicle_id: str
    capacity: int
    max_time: float
    start_x: float = 0.0
    start_y: float = 0.0
    end_x: float | None = None
    end_y: float | None = None

    def __post_init__(self) -> None:
        if not self.vehicle_id:
            raise ValueError("vehicle_id must not be empty")
        if self.capacity < 0:
            raise ValueError("capacity must be non-negative")
        if self.max_time < 0:
            raise ValueError("max_time must be non-negative")


@dataclass(frozen=True)
class VehicleRoute:
    vehicle_id: str
    stops: tuple[str, ...]
    total_distance: float
    total_time: float
    used_capacity: int


@dataclass(frozen=True)
class SelectionSummary:
    selected_ids: tuple[str, ...]
    total_cost: int
    total_value: float
    covered_cells: frozenset[str]
    uncovered_cell_value: float


def knapsack_select(candidates: list[DeicingCandidate], budget: int) -> list[DeicingCandidate]:
    """정해진 비용 예산 안에서 가치가 높은 제설 후보를 선택한다.

    0/1 Knapsack baseline이다. 커버리지 중복과 이동 거리는 고려하지 않으므로,
    설명하기 쉬운 비교 모델로 쓰기 좋다.
    """
    if budget < 0:
        raise ValueError("budget must be non-negative")

    n = len(candidates)
    dp = [[0.0] * (budget + 1) for _ in range(n + 1)]

    for i, candidate in enumerate(candidates, start=1):
        for cap in range(budget + 1):
            best = dp[i - 1][cap]
            if candidate.cost <= cap:
                best = max(best, dp[i - 1][cap - candidate.cost] + candidate.value)
            dp[i][cap] = best

    selected: list[DeicingCandidate] = []
    cap = budget
    for i in range(n, 0, -1):
        candidate = candidates[i - 1]
        if candidate.cost <= cap and dp[i][cap] == dp[i - 1][cap - candidate.cost] + candidate.value:
            selected.append(candidate)
            cap -= candidate.cost

    selected.reverse()
    return selected


def budgeted_maximum_coverage_select(
    candidates: list[DeicingCandidate],
    cell_values: dict[str, float],
    budget: int,
) -> list[DeicingCandidate]:
    """Greedy 방식의 Budgeted Maximum Coverage 대상 선정.

    각 후보는 위험 grid cell 집합을 커버한다. 이 알고리즘은 예산 안에서
    아직 덮이지 않은 cell 가치를 비용 대비 가장 많이 늘리는 후보를 반복 선택해
    같은 위험 구역을 중복 처리하는 문제를 줄인다.
    """
    if budget < 0:
        raise ValueError("budget must be non-negative")
    if any(value < 0 for value in cell_values.values()):
        raise ValueError("cell_values must be non-negative")

    remaining_budget = budget
    uncovered = set(cell_values)
    remaining = list(candidates)
    selected: list[DeicingCandidate] = []

    while remaining_budget > 0:
        best_candidate: DeicingCandidate | None = None
        best_key: tuple[float, float, int, str] | None = None

        for candidate in remaining:
            if candidate.cost > remaining_budget:
                continue
            newly_covered = candidate.covered_cells & uncovered
            gain = sum(cell_values[cell_id] for cell_id in newly_covered)
            if gain <= 0:
                continue
            key = (gain / candidate.cost, gain, -candidate.cost, candidate.candidate_id)
            if best_key is None or key > best_key:
                best_candidate = candidate
                best_key = key

        if best_candidate is None:
            break

        selected.append(best_candidate)
        remaining_budget -= best_candidate.cost
        uncovered -= best_candidate.covered_cells
        remaining = [candidate for candidate in remaining if candidate != best_candidate]

    return selected


def hybrid_bmc_knapsack_select(
    candidates: list[DeicingCandidate],
    cell_values: dict[str, float],
    budget: int,
    coverage_weight: float = 0.55,
    priority_weight: float = 0.45,
) -> list[DeicingCandidate]:
    """새 커버리지 증가량과 도로 자체 우선순위를 함께 보고 대상을 선택한다.

    다운로드한 대회 파이프라인도 이 아이디어를 사용한다. 순수 Knapsack은
    설명이 쉽지만 가까운 도로를 과하게 고를 수 있고, 순수 BMC는 도로 자체의
    위험 점수를 약하게 반영할 수 있다. 이 하이브리드 점수는 둘을 함께 유지한다.

    score = (coverage_weight * normalized_new_coverage
             + priority_weight * normalized_candidate_value) / cost
    """
    if budget < 0:
        raise ValueError("budget must be non-negative")
    if coverage_weight < 0 or priority_weight < 0:
        raise ValueError("weights must be non-negative")
    if coverage_weight == 0 and priority_weight == 0:
        raise ValueError("at least one weight must be positive")
    if any(value < 0 for value in cell_values.values()):
        raise ValueError("cell_values must be non-negative")

    remaining_budget = budget
    uncovered = set(cell_values)
    remaining = list(candidates)
    selected: list[DeicingCandidate] = []

    max_candidate_value = max((candidate.value for candidate in candidates), default=1.0)
    max_candidate_value = max(max_candidate_value, 1e-9)
    max_initial_coverage = max(
        (sum(cell_values[cell_id] for cell_id in candidate.covered_cells & uncovered) for candidate in candidates),
        default=1.0,
    )
    max_initial_coverage = max(max_initial_coverage, 1e-9)

    while remaining_budget > 0:
        best_candidate: DeicingCandidate | None = None
        best_key: tuple[float, float, float, int, str] | None = None

        for candidate in remaining:
            if candidate.cost > remaining_budget:
                continue
            coverage_gain = sum(cell_values[cell_id] for cell_id in candidate.covered_cells & uncovered)
            coverage_norm = coverage_gain / max_initial_coverage
            priority_norm = candidate.value / max_candidate_value
            hybrid_value = coverage_weight * coverage_norm + priority_weight * priority_norm
            if hybrid_value <= 0:
                continue
            score = hybrid_value / candidate.cost
            key = (score, coverage_gain, candidate.value, -candidate.cost, candidate.candidate_id)
            if best_key is None or key > best_key:
                best_candidate = candidate
                best_key = key

        if best_candidate is None:
            break

        selected.append(best_candidate)
        remaining_budget -= best_candidate.cost
        uncovered -= best_candidate.covered_cells
        remaining = [candidate for candidate in remaining if candidate != best_candidate]

    return selected


def summarize_selection(
    selected: list[DeicingCandidate],
    cell_values: dict[str, float],
) -> SelectionSummary:
    """선택된 제설 대상 묶음의 핵심 지표를 요약해 반환한다."""
    covered_cells: set[str] = set()
    for candidate in selected:
        covered_cells.update(candidate.covered_cells)

    uncovered_cell_value = sum(
        value for cell_id, value in cell_values.items()
        if cell_id not in covered_cells
    )

    return SelectionSummary(
        selected_ids=tuple(candidate.candidate_id for candidate in selected),
        total_cost=sum(candidate.cost for candidate in selected),
        total_value=sum(candidate.value for candidate in selected),
        covered_cells=frozenset(covered_cells),
        uncovered_cell_value=uncovered_cell_value,
    )


def _end_x(vehicle: RoutingVehicle) -> float:
    return vehicle.start_x if vehicle.end_x is None else vehicle.end_x


def _end_y(vehicle: RoutingVehicle) -> float:
    return vehicle.start_y if vehicle.end_y is None else vehicle.end_y


def _distance(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


def plan_vrp_routes(
    candidates: list[DeicingCandidate],
    vehicles: list[RoutingVehicle],
    value_weight: float = 0.15,
) -> list[VehicleRoute]:
    """선택된 후보를 단순 다중 차량 nearest-neighbor 휴리스틱으로 경로화한다.

    MVP용으로 의도적으로 가볍게 만든 방식이다. 차량 용량과 최대 작업 시간을
    지키면서 가까운 후보를 탐욕적으로 추가한다. 거리 차이가 비슷할 때는
    작은 가치 가중치 덕분에 고위험 도로가 우선될 수 있다.
    """
    if value_weight < 0:
        raise ValueError("value_weight must be non-negative")

    unassigned = {candidate.candidate_id: candidate for candidate in candidates}
    routes: list[VehicleRoute] = []

    for vehicle in vehicles:
        stops: list[str] = []
        used_capacity = 0
        total_distance = 0.0
        total_time = 0.0
        cur_x = vehicle.start_x
        cur_y = vehicle.start_y

        while unassigned:
            feasible: list[tuple[float, float, str, DeicingCandidate]] = []
            for candidate in unassigned.values():
                if used_capacity + candidate.demand > vehicle.capacity:
                    continue
                travel = _distance(cur_x, cur_y, candidate.x, candidate.y)
                return_to_depot = _distance(candidate.x, candidate.y, _end_x(vehicle), _end_y(vehicle))
                projected_time = total_time + travel + candidate.service_time + return_to_depot
                if projected_time <= vehicle.max_time:
                    route_score = travel - value_weight * candidate.value
                    feasible.append((route_score, travel, candidate.candidate_id, candidate))

            if not feasible:
                break

            _, travel, candidate_id, candidate = min(feasible)
            stops.append(candidate_id)
            used_capacity += candidate.demand
            total_distance += travel
            total_time += travel + candidate.service_time
            cur_x = candidate.x
            cur_y = candidate.y
            del unassigned[candidate_id]

        return_distance = _distance(cur_x, cur_y, _end_x(vehicle), _end_y(vehicle))
        total_distance += return_distance
        total_time += return_distance
        routes.append(
            VehicleRoute(
                vehicle_id=vehicle.vehicle_id,
                stops=tuple(stops),
                total_distance=total_distance,
                total_time=total_time,
                used_capacity=used_capacity,
            )
        )

    return routes


def plan_deicing_targets_and_routes(
    candidates: list[DeicingCandidate],
    cell_values: dict[str, float],
    budget: int,
    vehicles: list[RoutingVehicle],
) -> tuple[list[DeicingCandidate], list[VehicleRoute]]:
    """Hybrid BMC/Knapsack으로 제설 대상을 고르고 VRP로 경로화한다."""
    selected = hybrid_bmc_knapsack_select(candidates, cell_values, budget)
    routes = plan_vrp_routes(selected, vehicles)
    return selected, routes


@dataclass(frozen=True)
class Job:
    job_id: str
    demand: int
    priority: int


@dataclass(frozen=True)
class Vehicle:
    vehicle_id: str
    capacity: int


@dataclass(frozen=True)
class Allocation:
    vehicle_id: str
    job_id: str
    units: int


def greedy_allocate(jobs: list[Job], vehicles: list[Vehicle]) -> list[Allocation]:
    """Greedy baseline: 우선순위가 높은 작업부터 큰 용량 차량에 배정한다."""
    remaining = {job.job_id: job.demand for job in jobs}
    jobs_by_priority = sorted(jobs, key=lambda j: (-j.priority, -j.demand, j.job_id))
    vehicles_by_capacity = sorted(vehicles, key=lambda v: (-v.capacity, v.vehicle_id))

    allocations: list[Allocation] = []
    for vehicle in vehicles_by_capacity:
        free = vehicle.capacity
        for job in jobs_by_priority:
            if free <= 0:
                break
            need = remaining[job.job_id]
            if need <= 0:
                continue
            units = min(free, need)
            allocations.append(Allocation(vehicle.vehicle_id, job.job_id, units))
            remaining[job.job_id] -= units
            free -= units

    return allocations
