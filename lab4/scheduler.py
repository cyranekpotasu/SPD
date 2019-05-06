from typing import List

import numpy as np

from lab4.heap import Heap, HeapObject
from lab4.job import Job


def schrage_algorithm(jobs: List[Job]) -> List[Job]:
    """Implementation of Schrage algorithm."""
    jobs = jobs.copy()
    time = min(job.preparation for job in jobs)
    perm = []
    ready_jobs = []
    while jobs or ready_jobs:
        if jobs:
            ready_jobs.extend(job for job in jobs if job.preparation <= time)
            jobs = [job for job in jobs if job not in ready_jobs]
        if not ready_jobs:
            time = min(job.preparation for job in jobs)
        else:
            longest_delivery = max(job.delivery for job in ready_jobs)
            longest_delivery_job = next(job for job in ready_jobs
                                        if job.delivery == longest_delivery)
            ready_jobs.pop(ready_jobs.index(longest_delivery_job))
            perm.append(longest_delivery_job)
            time += longest_delivery_job.execution
    return perm


def schrage_pmtn(jobs: List[Job]) -> int:
    """Implementation of Schrage with interrupts. Returns total makespan."""
    makespan = 0
    jobs = jobs.copy()
    time = min(job.preparation for job in jobs)
    ready_jobs = []
    current_delivery = float('inf')
    current_job = jobs[0]
    while jobs or ready_jobs:
        while jobs and min(job.preparation for job in jobs) <= time:
            shortest_prep_job = next(job for job in jobs
                                     if job.preparation == min(job.preparation
                                                               for job in jobs))
            ready_jobs.append(shortest_prep_job)
            jobs = [job for job in jobs if job not in ready_jobs]
            if shortest_prep_job.delivery > current_delivery:
                current_job.execution = time - shortest_prep_job.preparation
                time = shortest_prep_job.preparation
                if current_job.execution > 0:
                    ready_jobs.append(current_job)
        if not ready_jobs:
            time = min(job.preparation for job in jobs)
        else:
            longest_delivery = max(job.delivery for job in ready_jobs)
            longest_delivery_job = next(job for job in ready_jobs
                                        if job.delivery == longest_delivery)
            ready_jobs.pop(ready_jobs.index(longest_delivery_job))
            current_job = longest_delivery_job
            current_delivery = current_job.delivery
            time += longest_delivery_job.execution
            makespan = max(makespan, time + longest_delivery_job.delivery)
    return makespan


def schrage_heaps(jobs: List[Job]) -> List[Job]:
    """Implementation of Schrage algorithm using heaps."""
    jobs_heap = Heap([HeapObject(job, job.preparation) for job in jobs])
    time = jobs_heap.peek().preparation
    perm = []
    ready_jobs = Heap()
    while jobs_heap or ready_jobs:
        while jobs_heap and jobs_heap.peek().preparation <= time:
            shortest_prep_job = jobs_heap.pop()
            ready_jobs.push(shortest_prep_job, -shortest_prep_job.delivery)
        if not ready_jobs:
            time = jobs_heap.peek().preparation
        else:
            longest_delivery_job = ready_jobs.pop()
            perm.append(longest_delivery_job)
            time += longest_delivery_job.execution
    return perm


def schrage_pmtn_heaps(jobs: List[Job]) -> List[Job]:
    """Implementation of Schrage algorithm with interrupts using heaps."""
    jobs_heap = Heap([HeapObject(job, job.preparation) for job in jobs])
    time = jobs_heap.peek().preparation
    makespan = 0
    current_delivery = float('inf')
    current_job = jobs_heap.peek()
    ready_jobs = Heap()
    while jobs_heap or ready_jobs:
        while jobs_heap and jobs_heap.peek().preparation <= time:
            shortest_prep_job = jobs_heap.pop()
            ready_jobs.push(shortest_prep_job, -shortest_prep_job.delivery)
            if shortest_prep_job.delivery > current_delivery:
                current_job.execution = time - shortest_prep_job.preparation
                time = shortest_prep_job.preparation
                if current_job.execution > 0:
                    ready_jobs.push(current_job, -current_job.delivery)
        if not ready_jobs:
            time = jobs_heap.peek().preparation
        else:
            longest_delivery_job = ready_jobs.pop()
            current_job = longest_delivery_job
            current_delivery = current_job.delivery
            time += longest_delivery_job.execution
            makespan = max(makespan, time + longest_delivery_job.delivery)
    return makespan


def sim_annealing(jobs, initial_temp=100000, cooling_factor=0.95,
                  iterations=10000, accept_equal=True):
    """Simulated annealing implementation"""
    jobs = jobs.copy()
    for i in range(iterations):
        job_new = random_insert(jobs)
        makespan = compute_makespan(jobs)
        next_makespan = compute_makespan(job_new)
        if not accept_equal and makespan == next_makespan:
            continue
        if (np.random.uniform(0, 1)
                < np.exp((makespan - next_makespan) / initial_temp)):
            jobs = job_new
            initial_temp *= cooling_factor
    solution_order = [job.id for job in jobs]
    return compute_makespan(jobs), solution_order


def random_insert(source_list: List[Job]):
    """Pop random item from a list and insert in random place."""
    result_list = source_list.copy()
    tmp = result_list.pop(np.random.randint(len(result_list)))
    result_list.insert(np.random.randint(len(result_list)), tmp)
    return result_list


def compute_makespan(permutation: List[Job]):
    """Compute makespan for given permutation of jobs."""
    time = 0
    makespan = 0
    for job in permutation:
        job_makespan = max(time, job.preparation) + job.execution + job.delivery
        if job.preparation > time:
            time += job.preparation - time + job.execution
        else:
            time += job.execution
        makespan = max(makespan, job_makespan)
    return makespan
