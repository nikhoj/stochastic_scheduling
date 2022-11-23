import random
import math

import numpy as np


class Job:
    def __init__(self, id):
        self.id = id
        self.mean = random.randint(50,300)
        self.var = random.randint(1,15)
        self.std = np.sqrt(self.var)

        # self.mean = round((1 + xi) ** int(math.log(self.mean, (1 + xi))))
        # self.var = round((1 + xi) ** int(math.log(self.var, (1 + xi))))


class Machine:
    def __init__(self, id):
        self.id = id


class Schedule:

    def __init__(self, xi, machines, jobs, T):

        self.machines = machines
        self.jobs = jobs
        self.xi = xi
        self.T = T

    def sum_of_variance(self):

        sum = 0
        for job in self.jobs:
            sum += job.var

        return sum

    def final_possible_list(self):  ## dd --> due date
        alist = [0, self.T]
        i = 0
        while (self.T + (1 + self.xi) ** i) <= 1.25 * self.T and (self.T - (1 + self.xi) ** i) >= self.T / 1.25:
            alist.append(self.T + (1 + self.xi) ** i)
            alist.append(self.T - (1 + self.xi) ** i)
            i += 1
        # print(min(alist), max(alist))

        return sorted(alist)

    def final_possible_var(self):
        alist = [0]
        i = 0
        while ((1 + self.xi) ** i) <= (self.sum_of_variance()/len(self.machines))*2 :
            alist.append((1 + self.xi) ** i)
            i += 1

        # print(min(alist), max(alist))
        return sorted(alist)
