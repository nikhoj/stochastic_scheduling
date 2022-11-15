import random

import numpy as np
from data_structure_FPT import Job, Machine
import scipy.stats as st
import time
import gurobipy as gp
from gurobipy import GRB


# Algorithm start here
def algorithm_OPT(m, n, delta):
    # Initialization
    machines = []
    jobs = []
    # delta = 50 * (n / m)
    random.seed(0)

    for i in range(m):
        machines.append(Machine(i))
    for j in range(n):
        jobs.append(Job(j))

    # algorithm starts here
    model = gp.Model("stochastic_scheduling-OPT")

    # variable declaration
    x = model.addVars(machines, jobs, vtype=GRB.BINARY, name="x")
    gmu = model.addVars(machines, name="gmu")
    gstd = model.addVars(machines, name="gvar")

    # Set objective
    get_mul = 1
    for i in machines:

        get_mul *= st.norm.cdf((delta - gmu[i])/5)
    obj_val = get_mul

    model.setObjective(obj_val, GRB.MAXIMIZE)

    # Add constraint:
    model.addConstrs((gp.quicksum(jobs[j.id].mean * x[i, j] for j in range(n)) <= gmu[i] for i in range(m)), "c0")
    model.addConstrs((gp.quicksum(jobs[j.id].var * x[i, j] for j in range(n)) <= gvar[i] for i in range(m)), "c1")
    model.addConstrs((gp.quicksum(x[i, j] for i in range(m)) == 1 for j in range(n)), "c2")
    model.addConstrs((gmu[i] <= 300 for i in range(m)), "c3")
    model.addConstrs((gstd[i]**2 <= 100 for i in range(n)), "c4")
    #model.addConstrs((gstd[i]**2 > 1 for i in range(n)), "c5")

    # Optimize model
    model.optimize()

    for v in m.getVars():
        print('%s %g' % (v.VarName, v.X))

    print('Obj: %g' % model.ObjVal)


algorithm_OPT(2, 5, 300)
