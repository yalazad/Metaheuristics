import numpy as np
import matplotlib.pyplot as plt
from ypstruct import structure 
import eggcrate_ga
import math

def eggcrate(x):
    return sum(x[:-1]**2.0 + x[1:]**2.0 + 25 * (math.sin(x[:-1])**2.0 + math.sin(x[1:])**2.0))

#Problem definition
problem = structure()
problem.costfunc = eggcrate
problem.num_of_vars = 2
problem.var_lower_bound = -2 * math.pi
problem.var_upper_bound = 2 * math.pi

#GA Parameters
params = structure()
params.max_gen = 30 # Max no. of generations
params.pop_num = 28 # Population number
params.prpn_children = 1 # Proportion of children
params.gamma = 0.1 # Crossover parameter to expand search space
params.mu = 0.53 # Mutation variable
params.sigma = 0.3 # Mutation variable
params.beta = 1 # Variable for roulette wheel selection

#Run GA
out = eggcrate_ga.run(problem, params)

#Results
for idx, val in enumerate(out.accumulated):
    plt.plot(out.accumulated[idx],label="Run " + str(idx+1), linewidth=1)
    #plt.semilogy(out.accumulated[idx],label="Run " + str(idx+1), linewidth=1)

#plt.semilogy(out.bestcost)
plt.xlim(0, params.max_gen)
plt.ylim(0, 20)
plt.xlabel('No. of Generations')
plt.ylabel('Objective Function')
plt.title('Genetic Algorithm (GA) for Eggcrate function')
plt.grid(True)
plt.legend(loc='best')
plt.show()