from typing import List

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
