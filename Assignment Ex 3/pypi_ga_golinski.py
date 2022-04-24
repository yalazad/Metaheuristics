import numpy as np
from geneticalgorithm import geneticalgorithm as ga

def f(x):
    #pen=0
    #if X[0]+X[1]<2:
        #pen=500+1000*(2-X[0]-X[1])
    #return np.sum(X)+pen
   # return 0.7854*x[0]*x[1]**2*( 3.3333*x[2]**2 + 14.9334*x[2] - 43.0934) \
    #    -1.5079*x[0]*(x[5]**2 + x[6]**2) + 7.477*(x[5]**3 + x[6]**3) + 0.7854*(x[3]*x[5]**2 + x[4]*x[6]**2)
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

    #c = 250000 #penalty parameter
    c = 1000000 #penalty parameter

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

varbound=np.array([[2.6,3.6],[0.7,0.8],[17,28],[7.3,8.3],[7.3,8.3],[2.9,3.9],[5.0,5.5]])
#max_num_iter = 500
#pop size = 70
algorithm_param = {'max_num_iteration': 1500,\
                   'population_size':500,\
                   'mutation_probability':0.95,\
                   'elit_ratio': 0.3,\
                   'crossover_probability': 0.8,\
                   'parents_portion': 0.3,\
                   'crossover_type':'uniform',\
                   'max_iteration_without_improv':None}

model=ga(function=f,\
            dimension=7,\
            variable_type='real',\
            variable_boundaries=varbound,\
            algorithm_parameters=algorithm_param)

#model=ga(function=f,dimension=2,variable_type='real',variable_boundaries=varbound)

model.run()
