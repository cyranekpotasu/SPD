from copy import deepcopy
from typing import List, Tuple, Optional

import numpy as np

from heap import Heap, HeapObject
from job import Job


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
    jobs = deepcopy(jobs)
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


class Carlier:
    """Carlier algorithm scheduler."""
    def __init__(self, jobs: List[Job]):
        self.jobs = deepcopy(jobs)
        self.best_perm = self.jobs
        self.upper_bound = max(makespan_list(self.best_perm))
        self._queue = [self.jobs]
        self.nodes = 0

    def schedule(self):
        while self._queue:
            self.jobs = self._queue.pop(0)
            self._carlier_node()
            self.nodes += 1
        return self.best_perm

    def _carlier_node(self):
        self.jobs = schrage_heaps(self.jobs)
        cmax_list = makespan_list(self.jobs)
        makespan = max(cmax_list)
        if self.upper_bound > makespan:
            self.upper_bound = makespan
            self.best_perm = deepcopy(self.jobs)
        b = cmax_list.argmax()
        a = self._find_a(b, makespan)
        c = self._find_c(a, b)
        if c is None:
            return
        block = self.jobs[(c + 1):(b + 1)]
        block_params = find_block_params(block)
        preparation_backup = self.jobs[c].preparation
        self.jobs[c].preparation = max(self.jobs[c].preparation, block_params[0] + block_params[1])
        lower_bound = schrage_pmtn_heaps(self.jobs)
        lower_bound = max(
            sum(block_params),
            sum(find_block_params(block + [self.jobs[c]])),
            lower_bound
        )
        if lower_bound < self.upper_bound:
            self._queue.append(deepcopy(self.jobs))
        self.jobs[c].preparation = preparation_backup
        delivery_backup = self.jobs[c].delivery
        self.jobs[c].delivery = max(self.jobs[c].delivery, block_params[1] + block_params[2])
        lower_bound = schrage_pmtn_heaps(self.jobs)
        lower_bound = max(
            sum(block_params),
            sum(find_block_params(block + [self.jobs[c]])),
            lower_bound
        )
        if lower_bound < self.upper_bound:
            self._queue.append(deepcopy(self.jobs))
        self.jobs[c].delivery = delivery_backup
        return

    def _find_a(self, b_index: int, makespan: int) -> int:
        """Find 'a' job for Carlier algorithm."""
        p_sum = sum(job.execution for job in self.jobs[:b_index + 1])
        q_max = self.jobs[b_index].delivery
        for i, job in enumerate(self.jobs):
            if i > b_index:
                raise ValueError("Job 'a' not found.")
            if job.preparation + p_sum + q_max == makespan:
                return i
            p_sum -= job.execution

    def _find_c(self, a_index: int, b_index: int) -> Optional[int]:
        """Find 'c' job for Carlier algorithm."""
        greater_deliveries = [index for index, job
                              in enumerate(self.jobs[a_index:b_index])
                              if job.delivery < self.jobs[b_index].delivery]
        return greater_deliveries[-1] + a_index if greater_deliveries else None

    @staticmethod
    def _find_block_params(block: List[Job]):
        """Find R, P, Q parameters for critical path in Carlier's algorithm."""
        return (min(job.preparation for job in block), sum(job.execution for job in block),
                min(job.delivery for job in block))


def carlier(jobs: List[Job], best_perm=None):
    """Implementation of Carlier algorithm."""
    jobs = schrage_heaps(jobs)
    cmax_list = makespan_list(jobs)
    makespan = max(cmax_list)
    if best_perm is None:
        best_perm = deepcopy(jobs)
    upper_bound = max(makespan_list(best_perm))
    if upper_bound > makespan:
        upper_bound = makespan
        best_perm = deepcopy(jobs)
    b = cmax_list.argmax()
    a = find_a(jobs, b, makespan)
    c = find_c(jobs, a, b)
    if c is None:
        return best_perm
    block = jobs[(c + 1):(b + 1)]
    block_params = find_block_params(block)
    preparation_backup = jobs[c].preparation
    jobs[c].preparation = max(jobs[c].preparation, block_params[0] + block_params[1])
    lower_bound = schrage_pmtn_heaps(jobs)
    lower_bound = max(sum(block_params), sum(find_block_params(block + [jobs[c]])), lower_bound)
    if lower_bound < upper_bound:
        candidate = carlier(jobs, best_perm)
        if max(makespan_list(candidate)) < upper_bound:
            best_perm = candidate
            upper_bound = max(makespan_list(candidate))
    jobs[c].preparation = preparation_backup
    delivery_backup = jobs[c].delivery
    jobs[c].delivery = max(jobs[c].delivery, block_params[1] + block_params[2])
    lower_bound = schrage_pmtn(jobs)
    lower_bound = max(sum(block_params), sum(find_block_params(block + [jobs[c]])), lower_bound)
    if lower_bound < upper_bound:
        candidate = carlier(jobs, best_perm)
        if max(makespan_list(candidate)) < upper_bound:
            best_perm = candidate
            upper_bound = max(makespan_list(candidate))
    jobs[c].delivery = delivery_backup
    return best_perm


def find_a(jobs: List[Job], b_index: int, makespan: int) -> int:
    """Find 'a' job for Carlier algorithm."""
    p_sum = sum(job.execution for job in jobs[:b_index + 1])
    q_max = jobs[b_index].delivery
    for i, job in enumerate(jobs):
        if i > b_index:
            raise ValueError("Job 'a' not found.")
        if job.preparation + p_sum + q_max == makespan:
            return i
        p_sum -= job.execution


def find_c(jobs: List[Job], a_index: int, b_index: int) -> Optional[int]:
    """Find 'c' job for Carlier algorithm."""
    greater_deliveries = [index for index, job in enumerate(jobs[a_index:b_index])
                          if job.delivery < jobs[b_index].delivery]
    return greater_deliveries[-1] + a_index if greater_deliveries else None


def find_block_params(block: List[Job]):
    """Find R, P, Q parameters for critical path in Carlier's algorithm."""
    return (min(job.preparation for job in block), sum(job.execution for job in block),
            min(job.delivery for job in block))


def makespan_list(permutation: List[Job]) -> np.ndarray:
    time = 0
    makespans = []
    for job in permutation:
        job_makespan = max(time, job.preparation) + job.execution + job.delivery
        if job.preparation > time:
            time += job.preparation - time + job.execution
        else:
            time += job.execution
        makespans.append(job_makespan)
    return np.array(makespans)


def compute_makespan(permutation: List[Job]) -> int:
    """Compute makespan for given permutation of jobs."""
    makespans = makespan_list(permutation)
    return np.max(makespans)
