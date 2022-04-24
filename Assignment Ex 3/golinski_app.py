import numpy as np
import matplotlib.pyplot as plt
from ypstruct import structure 
import golinski_ga
import math

def golinski(x):
     return 0.7854*x[0]*x[1]**2*( 3.3333*x[2]**2 + 14.9334*x[2] - 43.0934) \
        -1.5079*x[0]*(x[5]**2 + x[6]**2) + 7.477*(x[5]**3 + x[6]**3) + 0.7854*(x[3]*x[5]**2 + x[4]*x[6]**2)

def golinski_aug(x):
    #Constraints
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

    c = 50000000 #penalty parameter

    retval = (0.7854*x[0]*x[1]**2*( 3.3333*x[2]**2 + 14.9334*x[2] - 43.0934) \
        -1.5079*x[0]*(x[5]**2 + x[6]**2) + 7.477*(x[5]**3 + x[6]**3) + 0.7854*(x[3]*x[5]**2 + x[4]*x[6]**2))\
            + c * ((max(0, g1)**2)\
                + (max(0, g2)**2)\
                    + (max(0, g3)**2)\
                        + (max(0,g4)**2)\
                            + (max(0,g5)**2)\
                                + (max(0,g6)**2)\
                                    + (max(0,g7)**2)\
                                        + (max(0,g8)**2)\
                                            + (max(0,g9)**2)\
                                                + (max(0,g10)**2)\
                                                    + (max(0,g11)**2))
    #print("Golinski Augmented, return value is:", retval)
    return retval

#Problem definition
problem = structure()
problem.costfunc = golinski_aug
problem.num_of_vars = 7
problem.var_lower_bound = [2.6,0.7,17,7.3,7.3,2.9,5.0]
problem.var_upper_bound = [3.6,0.8,28,8.3,8.3,3.9,5.5]


#GA Parameters
params = structure()
params.max_gen = 140 # Max no. of generations
params.pop_num = 50 # Population number
params.prpn_children = 1 # Proportion of children
params.gamma = 0.8 # Crossover parameter to expand search space
params.mu = 0.1 # Mutation variable
params.sigma = 0.1 # Mutation variable
params.beta = 1 # Variable for roulette wheel selection


#Run GA
out = golinski_ga.run(problem, params)

#Results
for idx, val in enumerate(out.accumulated):
    plt.plot(out.accumulated[idx],label="Run " + str(idx+1), linewidth=1)
    #plt.semilogy(out.accumulated[idx],label="Run " + str(idx+1), linewidth=1)

plt.xlim(0, params.max_gen)
plt.ylim(0,120000)
plt.xlabel('No. of Generations')
plt.ylabel('Objective Function')
plt.title('Golinski Speed Reducer Genetic Algorithm (GA)')
plt.grid(True)
plt.legend(loc='best')
plt.show()