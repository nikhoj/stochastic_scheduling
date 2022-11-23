import gurobipy as gp
import st as st
from gurobipy import GRB
import numpy as np
from data_structure_FPT import Job, Machine, Schedule
import scipy.stats as st
import time
import random
from scipy.stats import norm
import itertools
from brute_force import algorithm1

def solve_FPT(sd):
    random.seed(sd)
    max_likelihood = 0
    # Initialization

    m = 2
    n = 15
    machines = []
    jobs = []

    I = []
    J = []
    for i in range(m):
        machines.append(Machine(i))
        I.append(i)

    for j in range(n):
        jobs.append(Job(j))
        J.append(j)



    mu_total =  0
    var_total = 0
    for job in jobs:
        mu_total+=job.mean
        var_total+=job.var

    dd = mu_total/m + np.sqrt((var_total/m)*m) + np.random.randint(-5,5)
    OPT = algorithm1(machines, jobs, delta=dd)


    # print(I,J)
    # print(jobs[j].mean, jobs[j].var)
    FPT = Schedule(xi=.9, machines=machines, jobs=jobs, T=dd)
    mu_list = FPT.final_possible_list()
    var_list = FPT.final_possible_var()

    # config generate
    # [mu range for machine i, var range of machine i]
    config_mu = []
    config_var = []

    for mu_i in range(len(mu_list) - 1):
        config_mu.append(mu_list[mu_i:mu_i + 2])
    for var_i in range(len(var_list) - 1):
        config_var.append(var_list[var_i:var_i + 2])

    config_mu = list(itertools.combinations_with_replacement(config_mu, r=m))
    config_var = list(itertools.combinations_with_replacement(config_var, r=m))
    # b = config_mu[1][2][1]
    max_time = 300
    start_time_inside = time.time()

    for mu_range in config_mu:
        for var_range in config_var:

            if time.time() - start_time_inside <= max_time:
            # print(mu_range[-1], var_range[-1])

                env = gp.Env(empty=True)
                env.setParam("OutputFlag", 0)
                env.start()

                # Create a new model
                model = gp.Model("stochastic_feasibility", env)
                model.setParam('TimeLimit', 60)
                # variable declaration
                x = model.addVars(I, J, vtype=GRB.BINARY, name="x")

                # objective value

                # var_sum = []

                # for i in I:
                #    var_sum.append(gp.quicksum(x[i, j] * jobs[j].var for j in J))

                # obj = 1
                # model.setObjective(obj, GRB.MINIMIZE)

                # constraints
                model.addConstrs((gp.quicksum(jobs[j].mean * x[i, j] for j in J) <= mu_range[i][1] for i in I), name="c0")
                model.addConstrs((gp.quicksum(jobs[j].mean * x[i, j] for j in J) >= mu_range[i][0] for i in I), name='c1')

                model.addConstrs((gp.quicksum(jobs[j].var * x[i, j] for j in J) <= var_range[i][1] for i in I), name="c2")
                model.addConstrs((gp.quicksum(jobs[j].var * x[i, j] for j in J) >= var_range[i][0] for i in I), name='c3')

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
                    # print(model.Status)
                    if model.Status == GRB.OPTIMAL:
                        # print("I am here")

                        table = np.zeros((1, m * 3 + 1), dtype=float)
                        #for v in model.getVars():
                            # print('%s %g' % (v.VarName, v.X))
                        table[0, -1] = 1
                        for i in I:
                            for j in J:
                                table[0, i * 3] += x[i, j].X * jobs[j].mean
                                table[0, i * 3 + 1] += x[i, j].X * jobs[j].var

                                table[0, i * 3 + 2] = norm.cdf(
                                    (dd - table[0, i * 3]) / max(1, np.sqrt(table[0, i * 3 + 1])))

                            table[0, -1] *= table[0, i * 3 + 2]

                        likelihood = table[0,-1]
                        if max_likelihood < likelihood:
                            max_likelihood = likelihood
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
                        print("For, m = {}, n = {}, delta = {}, P(Cmax <= delta) = {}".format(m, n, dd, table[index, -1]))


                except gp.GurobiError as e:
                    print('Error code ' + str(e.errno) + ": " + str(e))

                except AttributeError:
                    print('Encountered an attribute error')

            else:
                return max_likelihood
    print("---------------------------------------------------")
    print("OPT - ALG : {}".format(np.absolute(max_likelihood-OPT)))

    return max_likelihood


start = time.time()
prob = solve_FPT(500)
print("---------------------------------------------------")
print("maximum likelihood, P(Cmax <= delta) = {}".format(prob))
print("Total time it takes : {} sec".format(time.time() - start))
print("---------------------End----------------------")
