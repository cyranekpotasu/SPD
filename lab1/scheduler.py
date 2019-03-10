from typing import Sequence, NamedTuple, List
from itertools import permutations


class Job(NamedTuple):
    """Tuple storing data about job - its ID and times needed to complete for each machine."""
    id: int
    times: Sequence[int]


class Scheduler:
    def __init__(self, jobs: Sequence[Job]) -> None:
        self.jobs = jobs

    def complete_review(self):
        """Find the optimal schedule using complete review algorithm."""
        perms = list(permutations(self.jobs))
        makespans = []
        for perm in perms:
            makespans.append(get_makespan(perm))

        best_makespan = min(makespans)

        return perms[makespans.index(best_makespan)], best_makespan


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
