import os

import numpy as np

from scheduler import Job, schrage_algorithm

if __name__ == '__main__':
    for path in os.listdir('data'):
        job_data = np.loadtxt(f'data/{path}', dtype=int, skiprows=1)
        jobs = [Job(job_id, *times) for job_id, times in enumerate(job_data, 1)]
        perm = schrage_algorithm(jobs)
        print(f'Perm for {path}: {perm}')
