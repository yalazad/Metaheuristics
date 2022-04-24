import numpy as np
import matplotlib.pyplot as plt
from ypstruct import structure 
import rosen_ga

def rosen(x):
    return 100.0*(x[1]-x[0]**2.0)**2.0 + (1-x[0])**2.0

#Problem definition
problem = structure()
problem.costfunc = rosen
problem.num_of_vars = 2
problem.var_lower_bound = -5
problem.var_upper_bound = 5

#GA Parameters
params = structure()
params.max_gen = 30 # Max no. of generations
params.pop_num = 60 # Population number
params.prpn_children = 1 # Proportion of children
params.gamma = 0.1 # Crossover parameter 
params.mu = 0.53 # Mutation variable
params.sigma = 0.3 # Mutation variable
params.beta = 1 # Variable for roulette wheel selection

#Run GA
out = rosen_ga.run(problem, params)

#Results
for idx, val in enumerate(out.accumulated):
    plt.plot(out.accumulated[idx],label="Run " + str(idx+1), linewidth=1)
    #plt.semilogy(out.accumulated[idx],label="Run " + str(idx+1), linewidth=1)

#plt.semilogy(out.bestcost)
plt.xlim(0, params.max_gen)
plt.ylim(0, 7)
plt.xlabel('No. of Generations')
plt.ylabel('Objective Function')
plt.title('Genetic Algorithm (GA) for Rosenbrock Function')
plt.grid(True)
plt.legend(loc='best')
plt.show()