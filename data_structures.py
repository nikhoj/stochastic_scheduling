import random


class Job:
    def __init__(self, id):
        self.id = id
        self.mean = random.normalvariate(50,5)
        self.var = random.normalvariate(5,1)


class Machine:
    def __init__(self, id):
        self.id = id
