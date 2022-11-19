import random

import numpy as np
from data_structure_FPT import Job, Machine
import scipy.stats as st
import time
import gurobipy as gp
from gurobipy import GRB


# Algorithm start here
def algorithm1(m, n, delta, sd=0):
    # Initialization
    machines = []
    jobs = []
    # delta = 50 * (n / m)
    random.seed(sd)

    for i in range(m):
        machines.append(Machine(i))
    for j in range(n):
        jobs.append(Job(j))

    trace_table = np.zeros((1, 3 * m + 2))  # last column is to calculate the objective function score
    # print(trace_table[0])

    for j in range(1, n + 1):
        last_step_trace_table = trace_table[trace_table[:, 0] == j - 1]
        # l = len(last_step_trace_table)
        # for row in range(l):
        #    state = last_step_trace_table[row].copy()

        for i in range(1, 3 * m + 1, 3):
            current_step_trace_table = trace_table.copy()
            # current_step_trace_table = state.copy()
            current_step_trace_table[:, 0] = j
            current_step_trace_table[:, i] += jobs[j - 1].mean
            current_step_trace_table[:, i + 1] += jobs[j - 1].var

            trace_table = np.concatenate((trace_table, current_step_trace_table))
        trace_table = trace_table[trace_table[:, 0] == j]

    trace_table = trace_table[trace_table[:, 0] == n]
    for i in range(1, 3 * m + 1, 3):
        trace_table[:, i + 2] += st.norm.cdf((delta - trace_table[:, i]) / np.sqrt(trace_table[:, i + 1]))

    trace_table[:, -1] = trace_table[:, 3]
    for i in range(6, 3 * m + 1, 3):
        trace_table[:, -1] *= trace_table[:, i]
    # trace_table[:,-1] = st.norm.cdf(trace_table[:,-1])

    index = np.argmax(trace_table[:, -1])

    print("For, m = {}, n = {}, delta = {}, P(Cmax <= delta) = {}".format(m, n, delta, trace_table[index, -1]))
    return trace_table[index, -1]


start_time = time.time()
x = algorithm1(2, 10, 300)
end_time = time.time()
print("Running time : {}".format(end_time - start_time))
