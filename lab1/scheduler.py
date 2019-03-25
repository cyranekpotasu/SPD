from typing import Sequence, NamedTuple, List, Generator, Optional
from itertools import permutations
import numpy as np


class Job(NamedTuple):
    """Tuple storing data about job - its ID and times needed to complete for each machine."""
    id: int
    times: Sequence[int]

    def __eq__(self, other):
        return self.id == other.id


class Scheduler:
    def __init__(self, jobs: List[Job]) -> None:
        self.jobs = jobs

    def complete_review(self):
        """Find the optimal schedule using complete review algorithm."""
        perms = permutations(self.jobs)
        best_makespan = float('inf')
        best_permutation = None
        for perm in perms:
            makespan = get_makespan(perm)
            if makespan < best_makespan:
                best_makespan = makespan
                best_permutation = perm

        return best_permutation, best_makespan

    def johnsons_algorithm(self):
        """Find the optimal schedule using Johnson's algorithm."""
        machines_count = max(len(job.times) for job in self.jobs)
        if machines_count == 3:
            order = self._johnsons_three_machines(self.jobs.copy())
        else:
            order = self._johnsons_two_machines(self.jobs.copy())
        makespan = get_makespan(order)
        return order, makespan

    @staticmethod
    def _johnsons_two_machines(jobs: List[Job]):
        begin_list = []
        end_list = []

        while jobs:
            min_time = min(time for job in jobs for time in job.times)
            shortest_jobs = [job for job in jobs if min_time in job.times]
            job = shortest_jobs[0]
            job_index = jobs.index(job)
            machine_index = job.times.index(min_time)
            if machine_index == 0:
                begin_list.append(jobs.pop(job_index))
            else:
                end_list.insert(0, jobs.pop(job_index))

        return begin_list + end_list

    def _johnsons_three_machines(self, jobs):
        virtual_tasks = [
            Job(job.id, (job.times[0] + job.times[1], job.times[1] + job.times[2]))
            for job in jobs
        ]
        return self._johnsons_two_machines(virtual_tasks)

    def sorted_by_weight(self, jobs: List[Job]):
        weights = np.array([[job.id, (np.sum(job.times))] for job in jobs])
        sorted_weights = weights[weights[:, 1].argsort()][::-1]

        sorted_jobs = [jobs[index] for index, _ in sorted_weights]
        return sorted_jobs

    def find_insert_position(self, current_solution: List[Job],
                             job: Job) -> int:
        """Find position for which new makespan will be the shortest."""
        shortest_makespan = float("inf")
        best_position = None
        for i in range(len(current_solution) + 1):
            new_solution = current_solution[:]
            new_solution.insert(i, job)
            makespan = get_makespan(new_solution)
            if makespan < shortest_makespan:
                shortest_makespan = makespan
                best_position = i
        return best_position

    def neh_algorithm(self, improvement=0):
        sorted_jobs = self.sorted_by_weight(self.jobs.copy())
        solution = [self.jobs[sorted_jobs.pop(0).id]]

        for job in sorted_jobs:
            position = self.find_insert_position(solution, job)
            solution.insert(position, job)
            if improvement == 1:
                max_machine_time_index = self.neh_ir1(solution, job)
                if max_machine_time_index is not None:
                    max_machine_time_job = solution.pop(max_machine_time_index)
                    position = self.find_insert_position(solution, max_machine_time_job)
                    solution.insert(position, max_machine_time_job)

        solution_order = [job.id + 1 for job in solution]
        return get_makespan(solution), solution_order

    def neh_ir1(self, current_solution: List[Job], current_job: Job) -> Optional[int]:
        """Neh IR1 improvement implementation."""
        max_machine_time = 0
        max_index = None
        for index, job in enumerate(current_solution):
            job_max_time = max(job.times)
            if job_max_time > max_machine_time:
                max_machine_time = job_max_time
                max_index = index
        if current_solution[max_index] == current_job:
            return None
        return max_index


def get_makespan(job_list: Sequence[Job]) -> int:
    """Get total makespan of scheduled jobs."""
    times_arr = np.array([job.times for job in job_list])
    machine_times = count_job_times(times_arr)
    return machine_times[-1][-1]


def count_job_times(times_array: np.ndarray) -> np.ndarray:
    """Count completion times for each job for each machine,
    return as numpy array."""
    job_count, machine_count = times_array.shape
    makespan_array = np.pad(times_array, ((1, 0), (1, 0)), 'constant')
    for i in range(1, job_count + 1):
        for j in range(1, machine_count + 1):
            makespan_array[i, j] = max(makespan_array[i - 1, j],
                                       makespan_array[i, j - 1]) + makespan_array[i, j]
    return makespan_array
