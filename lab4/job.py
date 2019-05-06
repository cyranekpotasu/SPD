class Job:
    """Class representing job for RPQ problem."""
    def __init__(self, id: int, preparation: int,
                 execution: int, delivery: int):
        self.id = id
        self.preparation = preparation
        self.execution = execution
        self.delivery = delivery

    def __eq__(self, other):
        return self.id == other.id


