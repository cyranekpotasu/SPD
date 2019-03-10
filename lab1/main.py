from scheduler import Job, Scheduler

if __name__ == '__main__':
    jobs = [Job(0, (3, 2)), Job(1, (2, 8)), Job(2, (5, 1))]
    sched = Scheduler(jobs)
    print("Johnson's algorithm: ", sched.johnsons_algorithm())
    print("Complete review: ", sched.complete_review())
