from scheduler import Job, Scheduler
import numpy as np

if __name__ == '__main__':

    job_data = np.loadtxt('data.txt', dtype=int)
    jobs = [Job(job_id, times) for job_id, times in enumerate(job_data)]
    
    sched = Scheduler(jobs)
    makespan, job_order = sched.neh_algorihtm(jobs)
    print('Total makespan: ', makespan)
    print('Job order: ', job_order)
    makespan, job_order = sched.neh_algorihtm(jobs, improvement=1)
    print('Total makespan: ', makespan)
    print('Job order: ', job_order)
