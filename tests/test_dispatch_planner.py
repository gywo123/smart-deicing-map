from src.dispatch_planner import Job, Vehicle, greedy_allocate, mcmf_allocate


def test_greedy_allocate_fills_by_priority():
    jobs = [Job("J1", demand=5, priority=3), Job("J2", demand=4, priority=1)]
    vehicles = [Vehicle("V1", capacity=6), Vehicle("V2", capacity=3)]

    allocations = greedy_allocate(jobs, vehicles)
    total_j1 = sum(a.units for a in allocations if a.job_id == "J1")
    total_j2 = sum(a.units for a in allocations if a.job_id == "J2")

    assert total_j1 == 5
    assert total_j2 == 4


def test_mcmf_allocate_respects_cost_and_demand():
    jobs = [Job("J1", demand=3, priority=2), Job("J2", demand=2, priority=1)]
    vehicles = [Vehicle("V1", capacity=3), Vehicle("V2", capacity=3)]
    cost_matrix = {
        ("V1", "J1"): 1,
        ("V1", "J2"): 6,
        ("V2", "J1"): 4,
        ("V2", "J2"): 1,
    }

    allocations = mcmf_allocate(jobs, vehicles, cost_matrix)
    alloc = {(a.vehicle_id, a.job_id): a.units for a in allocations}

    assert alloc[("V1", "J1")] == 3
    assert alloc[("V2", "J2")] == 2
