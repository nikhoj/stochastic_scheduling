import gurobipy as gp
import st as st
from gurobipy import GRB
import numpy as np
from data_structure_FPT import Job, Machine
import scipy.stats as st
import time
import random
from scipy.stats import norm


def solve_FPT(sd):
    random.seed(sd)

    # Initialization
    dd = 360
    m = 4
    n = 20
    machines = []
    jobs = []

    I = []
    J = []
    for i in range(m):
        machines.append(Machine(i))
        I.append(i)

        mu_min, mu_max = machines[i].final_possible_list(dd=dd)
        var_min, var_max = machines[i].final_possible_var()

    for j in range(n):
        jobs.append(Job(j))
        J.append(j)

        # print(jobs[j].mean, jobs[j].var)

    # Create a new model
    model = gp.Model("stochastic_feasibility")

    # variable declaration
    x = model.addVars(I, J, vtype=GRB.BINARY, name="x")

    # objective value

    var_sum = []

    for i in I:
        var_sum.append(gp.quicksum(x[i, j] * jobs[j].var for j in J))

    obj = 1
    # model.setObjective(obj, GRB.MINIMIZE)

    # constraints
    model.addConstrs((gp.quicksum(jobs[j].mean * x[i, j] for j in J) <= mu_max for i in I), name="c0")
    # model.addConstrs((gp.quicksum(jobs[j].mean * x[i, j] for j in J) >= mu_min for i in I), name='c1')

    model.addConstrs((gp.quicksum(jobs[j].var * x[i, j] for j in J) <= var_max for i in I), name="c2")
    # model.addConstrs((gp.quicksum(jobs[j].var * x[i, j] for j in J) >= var_min for i in I), name='c3')

    model.addConstr(gp.quicksum(gp.quicksum(x[i, j] for j in J) for i in I) == n, name='c4')
    model.addConstrs((gp.quicksum(x[i, j] for i in I) == 1 for j in J), name='c5')

    # Save problem
    model.write("output.mps")
    model.write("output.lp")

    try:
        """
        # Limit how many solutions to collect
        model.setParam(GRB.Param.PoolSolutions, 1024)
        # Limit the search space by setting a gap for the worst possible solution
        # that will be accepted
        # model.setParam(GRB.Param.PoolGap, 0.10)
        # do a systematic search for the k-best solutions
        model.setParam(GRB.Param.PoolSearchMode, 2)
        """

        # Optimize
        model.optimize()

        table = np.zeros((1, m * 3 + 1), dtype=float)
        for v in model.getVars():
            print('%s %g' % (v.VarName, v.X))
        table[0, -1] = 1
        for i in I:
            for j in J:
                table[0, i * 3] += x[i, j].X * jobs[j].mean
                table[0, i * 3 + 1] += x[i, j].X * jobs[j].var

                table[0, i * 3 + 2] = norm.cdf((dd - table[0, i * 3]) / max(1, np.sqrt(table[0, i * 3 + 1])))

            table[0, -1] *= table[0, i * 3 + 2]

        print(table)
        """
        # Print number of solutions stored
        nSolutions = model.SolCount
        print('Number of solutions found: ' + str(nSolutions))
        table = np.zeros((nSolutions, m * 3 + 1), dtype=float)
        # print fourth best set if available
        for k in range(nSolutions):
            model.setParam(GRB.Param.SolutionNumber, k)
            table[k, -1] = 1
            #print("solution number {}".format(k))
            for i in I:

                for j in J:
                    #print(x[i, j])

                    table[k, i * 3] += x[i, j].X * jobs[j].mean
                    table[k, i * 3 + 1] += x[i, j].X * jobs[j].var

                    table[k, i * 3 + 2] = norm.cdf((dd - table[k, i * 3]) / max(1, np.sqrt(table[ k, i * 3 + 1])))

                table[k, -1] *= table[k, i * 3 + 2]
            """
        index = np.argmax(table[:, -1])
        print(table[index, -1])

    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))

    except AttributeError:
        print('Encountered an attribute error')


solve_FPT(1)
print("---End---")
