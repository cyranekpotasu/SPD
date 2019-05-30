import os
import random

import numpy as np

from lab4.job import Job

from lab6.scheduler import solve_rpq_cp, solve_witi


def generate_jobs(count):
    for i in range(count):
        yield Job(i + 1, *(random.randint(0, 2000) for _ in range(3)))


if __name__ == '__main__':
    for path in os.listdir('data'):
        print(path)
        job_data = np.loadtxt(f'data/{path}', dtype=int, skiprows=1)
        jobs = [Job(job_id, *times) for job_id, times in enumerate(job_data)]
        makespan, permutation = solve_rpq_cp(jobs)
        print(f'Makespan: {makespan}')
        print(f'Permutation: {permutation}')

    print('WiTi scheduling: ')
    for path in os.listdir('data_witi'):
        print(path)
        job_data = np.loadtxt(f'data_witi/{path}', dtype=int)
        makespan = solve_witi(job_data)
        print(f'Makespan: {makespan}')

