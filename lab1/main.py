from scheduler import Job, Scheduler
import numpy as np

if __name__ == '__main__':

    i = np.loadtxt('data.txt', dtype=int)
    jobs = [Job(j,i[j]) for j in range(len(i))]
    
    sched = Scheduler(jobs)
    sched.neh_algorihtm(jobs)
    
    
    