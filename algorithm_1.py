import numpy as np
from data_structures import Job, Machine

# generate job and machine here

m = 2   #number of machines
n = 10  #number of jobs

machines = []
jobs = []

for i in range(m):
    machines.append(Machine(i))
for j in range(n):
    jobs.append(Job(j))

# Algorithm start here
