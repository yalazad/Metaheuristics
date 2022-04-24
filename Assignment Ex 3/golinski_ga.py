import numpy as np
import matplotlib.pyplot as plt
from ypstruct import structure 
from time import perf_counter

def run(problem, params):
    # Problem information
    costfunc = problem.costfunc
    num_of_vars = problem.num_of_vars
    var_lower_bound = problem.var_lower_bound
    var_upper_bound = problem.var_upper_bound
    
    # Parameters
    max_gen = params.max_gen
    pop_num = params.pop_num
    beta = params.beta
    prpn_children = params.prpn_children
    num_children = int(np.round(prpn_children*pop_num/2)*2)
    gamma = params.gamma
    mu = params.mu
    sigma = params.sigma
    
    # Empty individual Template
    empty_individual = structure()
    empty_individual.position = None
    empty_individual.cost = None
  
    # For plotting results
    accumulator = list()
    accumulated = np.array([]) 

    num_of_runs = 20

    # Monitor performance
    t1_start = perf_counter()

    # Main loop
    for run in range(num_of_runs):
        #print("Run no. is:", run)

        # Best solution ever found
        bestsol = empty_individual.deepcopy()
        bestsol.cost = np.inf

        #Initialise population
        pop = empty_individual.repeat(pop_num)
        for i in range(pop_num):
            pop[i].position = np.random.uniform(var_lower_bound,var_upper_bound,num_of_vars)
            pop[i].cost = problem.costfunc(pop[i].position)
            if pop[i].cost < bestsol.cost:
                bestsol = pop[i].deepcopy()

        # Best cost of iterations
        bestcost = np.empty(max_gen)

        for it in range(max_gen):

            #Roulette wheel
            costs = [x.cost for x in pop]
            avg_cost = np.mean(costs)
            if avg_cost !=0:
                costs = costs/avg_cost
            probs = np.exp(-beta*costs)

            popc = []
            
            for _ in range(num_children//2):

                # Select parents
                #q = np.random.permutation(npop)
                #p1 = pop[q[0]]
                #p2 = pop[q[1]]

                # Perform Roulette Wheel Selection
                p1 = pop[roulette_wheel_selection(probs)]
                p2 = pop[roulette_wheel_selection(probs)]
            
                # Perform crossover
                c1, c2 = crossover(p1, p2, gamma)

                # Perform mutation
                c1 = mutate(c1, mu, sigma)
                c2 = mutate(c1, mu, sigma)

                # Apply bounds
                apply_bound(c1, var_lower_bound, var_upper_bound)
                apply_bound(c2, var_lower_bound, var_upper_bound)

                # Evaluate First Offspring
                c1.cost = costfunc(c1.position)
                if c1.cost < bestsol.cost:
                    bestsol = c1.deepcopy()

                # Evaluate Second Offspring
                c2.cost = costfunc(c2.position)
                if c2.cost < bestsol.cost:
                    bestsol = c2.deepcopy()

                # Add offsprings to popc
                popc.append(c1)
                popc.append(c2)
                #print("C1 is:",c1, ", C2 is:", c2)

            # Merge, sort and select
            pop += popc
            pop = sorted(pop, key=lambda x: x.cost)
            pop = pop[0:pop_num]

            # Store best cost
            bestcost[it] = round(bestsol.cost,3)
            X1 = round(bestsol.position[0],3)
            X2 = round(bestsol.position[1],3)
            X3 = round(bestsol.position[2],3)
            X4 = round(bestsol.position[3],3)
            X5 = round(bestsol.position[4],3)
            X6 = round(bestsol.position[5],3)
            X7 = round(bestsol.position[6],3)


            # Show iteration information
            if it == 139:
                print("Run no. {}, :Generation {}:, Best Cost {}:, Min X1 is {}:, Min X2 is {}:, Min X3 is {}:, Min X4 is {}:, Min X5 is {}:, Min X6 is {}:, Min X7 is {}".\
                    format(run, it, bestcost[it],X1,X2,X3,X4,X5,X6,X7))

            #Results
            # And now store the optimization path
            accumulator.append(bestcost[it])

    t1_stop = perf_counter()    
    print("Elapsed time:", t1_stop - t1_start)#In fractional seconds
                 
    # Output

    out = structure()
    out.pop = pop
    out.bestsol = bestsol
    out.bestcost = bestcost

    accumulated = np.append(accumulated, np.array(accumulator))
    accumulated.shape = (num_of_runs, params.max_gen)
    out.accumulated = accumulated
 
    return out

def crossover(p1, p2, gamma = 0.1):

    c1 = p1.deepcopy()
    c2 = p2.deepcopy()
    alpha = np.random.uniform(-gamma, 1+gamma, *c1.position.shape)
    c1.position = alpha*p1.position + (1-alpha)*p2.position
    c2.position = alpha*p2.position + (1-alpha)*p1.position
    return c1,c2

def mutate(x, mu, sigma):
    y = x.deepcopy()
    flag = np.random.rand(*x.position.shape) <= mu
    ind = np.argwhere(flag)
    y.position[ind] += sigma*np.random.randn(*ind.shape)
    return y

def apply_bound(x, varmin, varmax):
    x.position = np.maximum(x.position, varmin)
    x.position = np.minimum(x.position, varmax)

def roulette_wheel_selection(p):
    c = np.cumsum(p)
    r = sum(p)*np.random.rand()
    ind = np.argwhere(r <= c)
    return ind[0][0]

def constraints(x):
    g1 = 27/(x[0]*x[1]**2*x[2]) -1
    g2 = 397.5/(x[0]*x[1]**2*x[2]**2) -1
    g3 = 1.93/(x[1]*x[2]*x[5]**4)*x[3]**3 -1
    g4 = 1.93/(x[1]*x[2]*x[6]**4)*x[4]**3 -1
    g5 = np.sqrt( (745*x[3]/x[1]/x[2])**2 + 16.9e6)/(110.0*x[5]**3) -1
    g6 = np.sqrt( (745*x[4]/x[1]/x[2])**2 + 157.5e6)/(85.0*x[6]**3) -1
    g7 = x[1]*x[2]/40 -1
    g8 = 5*x[1]/x[0] -1
    g9 = x[0]/(12*x[1]) -1
    g10 = (1.5*x[5] + 1.9)/x[3] -1
    g11 = (1.1*x[6] + 1.9)/x[4] -1
    
    return np.array([g1,g2,g3,g4,g5,g6,g7,g8,g9,g10,g11]) 