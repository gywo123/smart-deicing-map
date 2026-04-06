"""Dispatch planning algorithms: MCMF and Greedy baselines."""

from __future__ import annotations

from dataclasses import dataclass
import heapq


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


class _Edge:
    __slots__ = ("to", "rev", "cap", "cost")

    def __init__(self, to: int, rev: int, cap: int, cost: int) -> None:
        self.to = to
        self.rev = rev
        self.cap = cap
        self.cost = cost


class _MinCostMaxFlow:
    def __init__(self, n: int) -> None:
        self.n = n
        self.graph: list[list[_Edge]] = [[] for _ in range(n)]

    def add_edge(self, u: int, v: int, cap: int, cost: int) -> None:
        fwd = _Edge(v, len(self.graph[v]), cap, cost)
        rev = _Edge(u, len(self.graph[u]), 0, -cost)
        self.graph[u].append(fwd)
        self.graph[v].append(rev)

    def min_cost_max_flow(self, s: int, t: int) -> tuple[int, int]:
        n = self.n
        flow = 0
        cost = 0
        potential = [0] * n

        while True:
            dist = [10**18] * n
            prev_node = [-1] * n
            prev_edge = [-1] * n
            dist[s] = 0
            pq: list[tuple[int, int]] = [(0, s)]

            while pq:
                d, u = heapq.heappop(pq)
                if d != dist[u]:
                    continue
                for i, e in enumerate(self.graph[u]):
                    if e.cap <= 0:
                        continue
                    nd = d + e.cost + potential[u] - potential[e.to]
                    if nd < dist[e.to]:
                        dist[e.to] = nd
                        prev_node[e.to] = u
                        prev_edge[e.to] = i
                        heapq.heappush(pq, (nd, e.to))

            if dist[t] == 10**18:
                break

            for i in range(n):
                if dist[i] < 10**18:
                    potential[i] += dist[i]

            addf = 10**18
            v = t
            while v != s:
                u = prev_node[v]
                e = self.graph[u][prev_edge[v]]
                addf = min(addf, e.cap)
                v = u

            v = t
            while v != s:
                u = prev_node[v]
                e = self.graph[u][prev_edge[v]]
                e.cap -= addf
                self.graph[v][e.rev].cap += addf
                v = u

            flow += addf
            cost += addf * potential[t]

        return flow, cost


def mcmf_allocate(jobs: list[Job], vehicles: list[Vehicle], cost_matrix: dict[tuple[str, str], int]) -> list[Allocation]:
    """Min-Cost Max-Flow allocation.

    - source -> vehicles (capacity)
    - vehicles -> jobs (assignment unit cost)
    - jobs -> sink (demand)
    """
    n_v = len(vehicles)
    n_j = len(jobs)
    source = 0
    first_vehicle = 1
    first_job = first_vehicle + n_v
    sink = first_job + n_j

    mcmf = _MinCostMaxFlow(sink + 1)

    for i, vehicle in enumerate(vehicles):
        mcmf.add_edge(source, first_vehicle + i, vehicle.capacity, 0)

    for j, job in enumerate(jobs):
        mcmf.add_edge(first_job + j, sink, job.demand, 0)

    for i, vehicle in enumerate(vehicles):
        for j, job in enumerate(jobs):
            c = cost_matrix[(vehicle.vehicle_id, job.job_id)]
            # lower effective cost for higher priority
            eff_cost = c - (job.priority * 10)
            mcmf.add_edge(first_vehicle + i, first_job + j, vehicle.capacity, eff_cost)

    mcmf.min_cost_max_flow(source, sink)

    allocations: list[Allocation] = []
    for i, vehicle in enumerate(vehicles):
        u = first_vehicle + i
        for e in mcmf.graph[u]:
            if first_job <= e.to < sink:
                job_index = e.to - first_job
                rev_edge = mcmf.graph[e.to][e.rev]
                used = rev_edge.cap
                if used > 0:
                    allocations.append(
                        Allocation(vehicle.vehicle_id, jobs[job_index].job_id, used)
                    )

    return allocations
