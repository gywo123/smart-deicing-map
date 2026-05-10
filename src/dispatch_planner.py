"""Dispatch planning algorithms for deicing target selection and routing.

The primary MVP pipeline is:

1. select target roads with Knapsack or Budgeted Maximum Coverage
2. route selected targets with a simple VRP nearest-neighbor heuristic

The older Greedy allocation function remains as a lightweight assignment
baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math


@dataclass(frozen=True)
class DeicingCandidate:
    """A candidate road segment or grid cell cluster that can be treated."""

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
    """Vehicle constraints for VRP-style routing."""

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
    """Select high-value deicing candidates under a cost budget.

    This is a 0/1 Knapsack baseline. It ignores overlapping coverage and routing
    distance, so it is useful as an easy-to-explain comparison model.
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
    """Greedy Budgeted Maximum Coverage selection.

    Each candidate covers a set of risk grid cells. The algorithm repeatedly
    picks the affordable candidate with the largest uncovered cell value per
    cost, which reduces duplicated treatment of the same risk area.
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
    """Select targets with both new coverage gain and road priority.

    The downloaded contest pipeline uses this idea: pure Knapsack is easy to
    explain but can over-select nearby roads, while pure BMC can understate a
    road's own risk score. This hybrid score keeps both:

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
    """Return compact metrics for a selected treatment set."""
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
    """Route selected candidates with a simple multi-vehicle nearest-neighbor heuristic.

    This is intentionally lightweight for the MVP. It respects vehicle capacity
    and max_time, then greedily adds the best nearby stop. A small value bias
    lets high-risk roads win when distances are similar.
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
    """Select deicing targets with hybrid BMC/Knapsack and route them with VRP."""
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
    """Greedy baseline: allocate highest-priority jobs first with largest capacity vehicles."""
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

