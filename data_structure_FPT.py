import random
import math

import numpy as np


class Job:
    def __init__(self, id):


        self.id = id
        self.mean = random.normalvariate(50, 5)
        self.var = random.normalvariate(5, 1) ** 2
        self.std = np.sqrt(self.var)

        #self.mean = round((1 + xi) ** int(math.log(self.mean, (1 + xi))))
        #self.var = round((1 + xi) ** int(math.log(self.var, (1 + xi))))


class Machine:
    def __init__(self, id):

        self.id = id

    def final_possible_list(self, dd, xi=.70):  ## dd --> due date
        alist = [dd]
        for i in range(20):
            alist.append(dd + (1 + xi) ** i )
            alist.append(dd - (1 + xi) ** i )
        # print(min(alist), max(alist))
        return min(alist), max(alist)

    def final_possible_var(self, xi=.50):
        alist = []
        for i in range(30):
            alist.append((1 + xi) ** i)

        # print(min(alist), max(alist))
        return min(alist), max(alist)
