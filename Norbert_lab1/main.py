from scheduler import Job, Scheduler

if __name__ == '__main__':
    jobs = [Job(0, (3, 2)), Job(1, (2, 8)), Job(2, (5, 1))]
    sched = Scheduler(jobs)
    print('Two machines: ')
    print("Johnson's algorithm: ", sched.johnsons_algorithm())
    print("Complete review: ", sched.complete_review())

    jobs3 = [Job(0, (3, 2, 1)), Job(1, (2, 8, 4)), Job(2, (5, 1, 1))]
    sched = Scheduler(jobs3)
    print('Three machines: ')
    print("Johnson's algorithm: ", sched.johnsons_algorithm())
    print("Complete review: ", sched.complete_review())
    