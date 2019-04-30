from typing import NamedTuple, List


class Job(NamedTuple):
    """Class representing job for RPQ problem."""
    id: int
    preparation: int
    execution: int
    delivery: int

    def __eq__(self, other):
        return self.id == other.id


def schrage_algorithm(jobs: List[Job]):
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
    return [job.id for job in perm]


