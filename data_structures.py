import random
import math


class Job:
    def __init__(self, id, xi = .0834):
        self.id = id
        self.mean = random.normalvariate(50,5)
        self.var = random.normalvariate(5,1)**2

        #self.mean = round((1+xi)**int(math.log(self.mean,(1+xi))))
        #self.var = round((1+xi) ** int(math.log(self.var, (1+xi))))

class Machine:
    def __init__(self, id):
        self.id = id
