import numpy as np
from data_structures import Job, Machine


# Algorithm start here
def algorithm1(m, n):
    # Initialization
    machines = []
    jobs = []

    for i in range(m):
        machines.append(Machine(i))
    for j in range(n):
        jobs.append(Job(j))

    trace_table = np.zeros((1, 2 * m + 1))
    #print(trace_table[0])

    for j in range(1, n + 1):
        last_step_trace_table = trace_table[trace_table[:, 0] == j - 1]
        l = len(last_step_trace_table)
        for row in range(l):
            state = last_step_trace_table[row].copy()

            for i in range(1, 2 * m, 2):
                current_step_trace_table = state.copy()
                current_step_trace_table[0] = j
                current_step_trace_table[i] += jobs[j - 1].mean
                current_step_trace_table[i + 1] += jobs[j - 1].var
                current_step_trace_table = np.reshape(current_step_trace_table, (1,2*m+1))
                trace_table = np.concatenate((trace_table, current_step_trace_table))

    print(trace_table)


algorithm1(2, 10)
print("end")