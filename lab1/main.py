import os
import timeit

import numpy as np

from scheduler import Job, Scheduler


if __name__ == '__main__':

    # for path in os.listdir('data'):
    #     job_data = np.loadtxt(f'data/{path}', dtype=int, skiprows=1 )
    #     jobs = [Job(job_id, times) for job_id, times in enumerate(job_data)]

    #     print(path)
    #     sched = Scheduler(jobs)
    #     makespan, job_order = sched.neh_algorithm()
    #     print('Total makespan: ', makespan)
    #     print('Job order: ', job_order)
    #     makespan, job_order = sched.neh_algorithm(improvement=1)
    #     print('Total makespan: ', makespan)
    #     print('Job order: ', job_order)

    job_data = np.loadtxt('data/ta020.txt', dtype=int, skiprows=1)
    jobs = [Job(job_id, times) for job_id, times in enumerate(job_data)]
    sched = Scheduler(jobs)

    print('NEH time: ', timeit.timeit('sched.neh_algorithm()', globals=globals(), number=1))
    print('SA time: ', timeit.timeit('sched.sim_annealing()', globals=globals(), number=1))

    makespan, job_order = sched.neh_sorted_list()
    print('Total makespan: ', makespan)
    print('Job order: ', job_order)

    sched.jobs = sched.neh_algorithm()
    makespan, job_order = sched.sim_annealing()
    print('Total makespan: ', makespan)
    print('Job order: ', job_order)
    makespans = [sched.sim_annealing()[0] for _ in range(5)]
    print('Avg of 5 runs: ',
          sum(makespan for makespan in makespans) / len(makespans))
    makespans = [sched.sim_annealing(accept_equal=False)[0] for _ in range(5)]
    print('Avg of 5 runs (ignore equal cmaxes): ',
          sum(makespan for makespan in makespans) / len(makespans))
