from scheduler import Job, Scheduler
import numpy as np

if __name__ == '__main__':

    job_data = np.loadtxt('data.txt', dtype=int)
    jobs = [Job(job_id, times) for job_id, times in enumerate(job_data)]
    
    sched = Scheduler(jobs)
    sched.neh_algorihtm(jobs)
    
    
    