from typing import Sequence, NamedTuple, List
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

    def array_with_zeros(self, array):
        array_zero = [[0 for jobs in range(len(array[0]) + 1)] for i in range(len(array) + 1)]
        for i in range(len(array[0])):
            for j in range(len(array)):
                array_zero[j + 1][i + 1] = array[j][i]
        return array_zero

    def compile_timeline(self, array):
        """Compute makespan."""
        cmax_table = self.array_with_zeros(array)

        for i in range(len(cmax_table[0]) - 1):
            for j in range(len(cmax_table) - 1):
                cmax_table[j + 1][i + 1] = max(int(cmax_table[j + 1][i]), int(cmax_table[j][i + 1])) + int(
                    cmax_table[j + 1][i + 1])

        return cmax_table[len(cmax_table) - 1][len(cmax_table[0]) - 1]

    def max_value_id(self, array):
        """Sort jobs by sum of their machine times."""
        id = []
        x = array

        for i in array:
            max = np.argmax(array)
            array[max] = -1
            id.append(max)

        return id

    def neh_algorithm(self, array):
        job_weight = np.sum(array, axis=1)
        id_weight = self.max_value_id(job_weight)

        sort_array = array[id_weight]

        new = sort_array[range(2)]

        for i in range(len(sort_array)):
            if i > 0:
                new = sort_array[range(i + 1)]
                for j in range(len(new)):
                    if j > 0:
                        cmax1 = self.compile_timeline(new)

                        new[[len(new) - j - 1, len(new) - j]] = new[[len(new) - j, len(new) - j - 1]]
                        cmax2 = self.compile_timeline(new)

                        if cmax2 < cmax1:
                            sort_array = np.vstack([new, sort_array[i + 1:]])  # dopisywanie

        return self.compile_timeline(sort_array)
