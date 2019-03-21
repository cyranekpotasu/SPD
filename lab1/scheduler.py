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

    def sort_array(self,jobs: List[Job]):


        sort = []
        weight = np.array([[job.id, (np.sum(job.times))] for job in jobs])
        weight  = weight[weight[:,1].argsort()]

        for i in range(len(weight)):
            sort.append(jobs[weight[i][0]])
        return sort[::-1]

    def neh(self,jobs: List[Job] ):


        new = jobs.copy()
        for j in range(len(jobs)):
            if(j>0):

                cmax1 = get_makespan(new)
                x = new[len(jobs)-j]
                new[len(new)-j] = new[len(new)-1-j]
                new[len(new)-1-j] = x

                cmax2 = get_makespan(new)

                if(cmax2<cmax1):
                    jobs = new

        return jobs


    def neh_algorihtm(self,jobs: List[Job]):

        sort_weight = self.sort_array(self.jobs.copy())
        tmp = []
        tmp.append(jobs[sort_weight[0][0]])
        t = []

        for i in range(len(sort_weight)):
            if(i>0):
                tmp.append(jobs[sort_weight[i][0]])
                tmp = self.neh(tmp)

        for i in range(len(tmp)):
            t.append(tmp[i][0] + 1)

        print(get_makespan(tmp))
        print(t)



def get_makespan(job_list: Sequence[Job]) -> int:
    """Get total makespan of scheduled jobs."""
    timeline = compile_timeline(job_list)
    return timeline[-1][-1] + job_list[-1].times[-1]


def compile_timeline(job_list: Sequence[Job]) -> List[List[int]]:
    """Compile given job permutation, return matrix in which rows represent
    machine id and columns job id."""
    machines_count = max(len(job.times) for job in job_list)

    machine_times = [[] for _ in range(machines_count)]
    machine_times[0].append(0)

    jobs_iter = iter(job_list)
    first_job = next(jobs_iter)

    for machine_index in range(1, machines_count):
        machine_times[machine_index].append(
            machine_times[machine_index - 1][0] + first_job.times[machine_index - 1]
        )

    prev_job = first_job

    for job_index in range(1, len(job_list)):
        job = job_list[job_index]
        machine_times[0].append(machine_times[0][-1] + prev_job.times[0])

        for machine_index in range(1, machines_count):
            machine_times[machine_index].append(
                max(machine_times[machine_index - 1][job_index] + job.times[machine_index - 1],
                    machine_times[machine_index][job_index - 1] + prev_job.times[machine_index])
            )
        prev_job = job

    return machine_times