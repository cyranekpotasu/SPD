
from scheduler import Job, Scheduler
import numpy as np

if __name__ == '__main__':
    jobs = np.loadtxt('data.txt', skiprows=1, dtype=int)

    sched = Scheduler(jobs)
    print(sched.neh_algoritm(jobs))