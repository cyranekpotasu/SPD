from collections import namedtuple
from typing import List, Tuple, Optional

import numpy as np
from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model

from lab4.job import Job

JobShopTaskData = List[List[Tuple[int, int]]]

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


def solve_rpq_cp(jobs: List[Job]):
    """Solve RPQ problem using Constraint Programming."""
    model = cp_model.CpModel()
    upper_bound = int(sum(job.total for job in jobs))

    makespan = model.NewIntVar(0, upper_bound, 'makespan')

    intervals = []
    job_vars = {}
    for job in jobs:
        start_var = model.NewIntVar(0, upper_bound, f'start_{job.id}')
        end_var = model.NewIntVar(0, upper_bound, f'end_{job.id}')
        interval_var = model.NewIntervalVar(start_var, job.execution,
                                            end_var, f'interval_{job.id}')
        intervals.append(interval_var)
        job_vars[job.id] = ({'start': start_var, 'end': end_var})

    model.AddNoOverlap(intervals)

    for job in jobs:
        model.Add(job_vars[job.id]['start'] >= job.preparation)
        model.Add(job_vars[job.id]['end'] + job.delivery <= makespan)

    model.Minimize(makespan)

    solver = cp_model.CpSolver()
    solver.Solve(model)

    start_times = [(job.id, solver.Value(job_vars[job.id]['start'])) for job in jobs]

    permutation = [job_id for job_id, _ in sorted(start_times, key=lambda x: x[1])]

    return solver.ObjectiveValue(), permutation


def solve_witi(jobs_data: np.ndarray):
    """Solve WiTi problem using CP."""
    witi_job = namedtuple('witijob', 'duration weight end_time')
    jobs = [witi_job(*job) for job in jobs_data]

    model = cp_model.CpModel()

    total_makespan = int(sum(job.duration for job in jobs))

    intervals = []
    delays = []
    for i, job in enumerate(jobs):
        start_var = model.NewIntVar(0, total_makespan, f'job_{i}')
        end_var = model.NewIntVar(0, total_makespan, f'job_{i}')
        interval_var = model.NewIntervalVar(start_var, job.duration, end_var, f'interval_{i}')
        delay_var = model.NewIntVar(0, total_makespan, f'delay_{i}')

        model.Add(delay_var >= 0)
        model.Add(delay_var >= end_var - job.end_time)

        intervals.append(interval_var)
        delays.append(delay_var)

    model.AddNoOverlap(intervals)

    total_cost = sum(delays[i] * job.weight for i, job in enumerate(jobs))

    model.Minimize(total_cost)

    solver = cp_model.CpSolver()
    solver.Solve(model)

    return solver.ObjectiveValue()


def solve_job_shop(jobs_data: JobShopTaskData):
    """Solve job shop problem using CP."""
    model = cp_model.CpModel()

    machine_count = max(task[0] for job in jobs_data for task in job)

    upper_bound = sum(task[1] for job in jobs_data for task in job)


    all_tasks = {}
    for job_id, job in enumerate(jobs_data, 1):
        for task_id, task in enumerate(job, 1):
            start_var = model.NewIntVar(0, upper_bound, f'start_{job_id}_{task_id}')
            end_var = model.NewIntVar(0, upper_bound, f'end_{job_id}_{task_id}')
            interval_var = model.NewIntervalVar(start_var, task[1], end_var,
                                                f'interval_{job_id}_{task_id}')
            all_tasks[job_id, task_id] = {'start': start_var, 'end': end_var,
                                          'interval': interval_var}

    for machine_id in range(1, machine_count + 1):
        intervals = []
        for job_id, job in enumerate(jobs_data, 1):
            for task_id, task in enumerate(job, 1):
                if task[0] == machine_id:
                    intervals.append(all_tasks[job_id, task_id]['interval'])
        model.AddNoOverlap(intervals)

    for job_id, job in enumerate(jobs_data, 1):
        for task_id in range(1, len(job)):
            model.Add(all_tasks[job_id, task_id]['end']
                      <= all_tasks[job_id, task_id + 1]['start'])

    makespan = model.NewIntVar(0, upper_bound, 'makespan')

    model.AddMaxEquality(
        makespan,
        [all_tasks[job_id, len(job)]['end'] for job_id, job in enumerate(jobs_data, 1)]
    )

    model.Minimize(makespan)
    solver = cp_model.CpSolver()
    solver.Solve(model)

    return solver.ObjectiveValue()
