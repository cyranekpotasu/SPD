from typing import List, Tuple, Optional

from ortools.linear_solver import pywraplp

from lab4.job import Job


def solve_rpq(jobs: List[Job]):
    solver = pywraplp.Solver('rpq_mip', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
    max_value = 0
    for job in jobs:
        max_value += job.preparation + job.delivery + job.execution

    alfas = [
        [solver.IntVar(0, 1, f'alpha{j},{i}') for i in range(len(jobs))] for j in range(len(jobs))
    ]
    starts = [solver.IntVar(0, int(max_value), f'start{i}') for i in range(len(jobs))]

    cmax = solver.IntVar(0, int(max_value), 'cmax')

    for job in jobs:
        solver.Add(starts[job.id] >= job.preparation)

    for job in jobs:
        solver.Add(cmax >= starts[job.id] + job.execution + job.delivery)

    for i, job1 in enumerate(jobs):
        for job2 in jobs[i + 1:]:
            solver.Add(starts[job1.id] + job1.execution
                       <= starts[job2.id] + alfas[job1.id][job2.id] * max_value)
            solver.Add(starts[job2.id] + job2.execution
                       <= starts[job1.id] + alfas[job2.id][job1.id] * max_value)
            solver.Add(alfas[job1.id][job2.id] + alfas[job2.id][job1.id] == 1)

    solver.Minimize(cmax)
    result = solver.Solve()

    if result != solver.OPTIMAL:
        print("Solver didn't find optimal solution.")
    print(solver.Objective().Value())











