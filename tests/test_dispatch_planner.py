from src.dispatch_planner import (
    DeicingCandidate,
    Job,
    RoutingVehicle,
    Vehicle,
    budgeted_maximum_coverage_select,
    greedy_allocate,
    hybrid_bmc_knapsack_select,
    knapsack_select,
    plan_deicing_targets_and_routes,
    plan_vrp_routes,
    summarize_selection,
)


def test_greedy_allocate_fills_by_priority():
    jobs = [Job("J1", demand=5, priority=3), Job("J2", demand=4, priority=1)]
    vehicles = [Vehicle("V1", capacity=6), Vehicle("V2", capacity=3)]

    allocations = greedy_allocate(jobs, vehicles)
    total_j1 = sum(a.units for a in allocations if a.job_id == "J1")
    total_j2 = sum(a.units for a in allocations if a.job_id == "J2")

    assert total_j1 == 5
    assert total_j2 == 4


def test_knapsack_selects_highest_value_under_budget():
    candidates = [
        DeicingCandidate("A", cost=4, value=10),
        DeicingCandidate("B", cost=5, value=11),
        DeicingCandidate("C", cost=3, value=7),
    ]

    selected = knapsack_select(candidates, budget=7)

    assert [candidate.candidate_id for candidate in selected] == ["A", "C"]


def test_budgeted_maximum_coverage_avoids_duplicate_cells():
    candidates = [
        DeicingCandidate("A", cost=2, value=20, covered_cells=frozenset({"c1", "c2"})),
        DeicingCandidate("B", cost=1, value=10, covered_cells=frozenset({"c2"})),
        DeicingCandidate("C", cost=1, value=10, covered_cells=frozenset({"c3"})),
    ]
    cell_values = {"c1": 10, "c2": 10, "c3": 10}

    selected = budgeted_maximum_coverage_select(candidates, cell_values, budget=3)

    assert [candidate.candidate_id for candidate in selected] == ["A", "C"]


def test_hybrid_bmc_knapsack_balances_coverage_and_priority():
    candidates = [
        DeicingCandidate("A", cost=2, value=3, covered_cells=frozenset({"c1", "c2"})),
        DeicingCandidate("B", cost=2, value=10, covered_cells=frozenset({"c2"})),
        DeicingCandidate("C", cost=1, value=4, covered_cells=frozenset({"c3"})),
    ]
    cell_values = {"c1": 5, "c2": 5, "c3": 1}

    selected = hybrid_bmc_knapsack_select(candidates, cell_values, budget=3)
    summary = summarize_selection(selected, cell_values)

    assert [candidate.candidate_id for candidate in selected] == ["B", "C"]
    assert summary.selected_ids == ("B", "C")
    assert summary.total_cost == 3
    assert summary.covered_cells == frozenset({"c2", "c3"})


def test_plan_vrp_routes_respects_capacity_and_time():
    candidates = [
        DeicingCandidate("A", cost=1, value=10, x=0, y=1, service_time=1, demand=1),
        DeicingCandidate("B", cost=1, value=9, x=0, y=3, service_time=1, demand=1),
        DeicingCandidate("C", cost=1, value=8, x=10, y=10, service_time=1, demand=1),
    ]
    vehicles = [
        RoutingVehicle("V1", capacity=2, max_time=8, start_x=0, start_y=0),
        RoutingVehicle("V2", capacity=2, max_time=40, start_x=0, start_y=0),
    ]

    routes = plan_vrp_routes(candidates, vehicles)

    assert routes[0].stops == ("A", "B")
    assert routes[0].used_capacity == 2
    assert routes[1].stops == ("C",)


def test_plan_deicing_targets_and_routes_uses_bmc_then_vrp():
    candidates = [
        DeicingCandidate("A", cost=2, value=20, covered_cells=frozenset({"c1", "c2"}), x=0, y=1),
        DeicingCandidate("B", cost=1, value=10, covered_cells=frozenset({"c2"}), x=0, y=2),
        DeicingCandidate("C", cost=1, value=10, covered_cells=frozenset({"c3"}), x=0, y=3),
    ]
    cell_values = {"c1": 10, "c2": 10, "c3": 10}
    vehicles = [RoutingVehicle("V1", capacity=3, max_time=20)]

    selected, routes = plan_deicing_targets_and_routes(candidates, cell_values, budget=3, vehicles=vehicles)

    assert [candidate.candidate_id for candidate in selected] == ["A", "C"]
    assert routes[0].stops == ("A", "C")
