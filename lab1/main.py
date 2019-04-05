import os
import numpy as np

from scheduler import Job, Scheduler


if __name__ == '__main__':


    for path in os.listdir('data'):
        job_data = np.loadtxt(f'data/{path}', dtype=int, skiprows=1 )
        jobs = [Job(job_id, times) for job_id, times in enumerate(job_data)]
        sched = Scheduler(jobs)
        # print(path)
        # makespan, job_order = sched.neh_sorted_list()
        # # print("(",len(jobs),len(jobs[0].times),")")
        # #print('Total makespan NEH: ', makespan)
        
        # # print('Job order NEH: ', job_order)

        # f = sched.neh_algorithm()
        makespan, job_order = sched.sa(jobs)
        print(makespan)
        # print('Job order SA: ', job_order)

    
        # job_data = np.loadtxt(f'data/ta100.txt', dtype=int, skiprows=1 )
        # jobs = [Job(job_id, times) for job_id, times in enumerate(job_data)]
        # sched = Scheduler(jobs)
        
        # makespan, job_order = sched.neh_sorted_list()
        # # print('Total makespan: ', makespan)
        # # print('Job order: ', job_order)
        
        # f = sched.neh_algorithm()
        # makespan, job_order = sched.sa(f)
        # print(makespan)
        # print('Total makespan: ', makespan)
        # print('Job order: ', job_order)

   

