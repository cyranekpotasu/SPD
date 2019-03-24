from typing import Sequence, NamedTuple, List, Generator
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
        perms = list(permutations(self.jobs))
        makespans = []
        for perm in perms:
            makespans.append(get_makespan(perm))

        best_makespan = min(makespans)

        return perms[makespans.index(best_makespan)], best_makespan

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

    def neh(self, jobs: List[Job]):
        new = jobs.copy()
        for j in range(len(jobs)):
            if j > 0:
                cmax1 = get_makespan(new)
                x = new[len(jobs) - j]
                new[len(new) - j] = new[len(new) - 1 - j]
                new[len(new) - 1 - j] = x

                cmax2 = get_makespan(new)

                if cmax2 < cmax1:
                    jobs = new
        return jobs

    def neh_algorihtm(self, jobs: List[Job]):
        sort_weight = self.sorted_by_weight(self.jobs.copy())
        tmp = [jobs[sort_weight[0][0]]]
        t = []

        for i in range(len(sort_weight)):
            if i > 0:
                tmp.append(jobs[sort_weight[i][0]])
                tmp = self.neh(tmp)

        for i in range(len(tmp)):
            t.append(tmp[i][0] + 1)

        return get_makespan(tmp), t


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
