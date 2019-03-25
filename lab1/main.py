import os
import numpy as np

from scheduler import Job, Scheduler


if __name__ == '__main__':

    for path in os.listdir('data'):
        job_data = np.loadtxt(f'data/{path}', dtype=int, skiprows=1)
        jobs = [Job(job_id, times) for job_id, times in enumerate(job_data)]

        print(path)
        sched = Scheduler(jobs)
        makespan, job_order = sched.neh_algorithm()
        print('Total makespan: ', makespan)
        print('Job order: ', job_order)
        makespan, job_order = sched.neh_algorithm(improvement=1)
        print('Total makespan: ', makespan)
        print('Job order: ', job_order)
