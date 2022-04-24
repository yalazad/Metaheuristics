import numpy as np
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint
from time import perf_counter
import random
from scipy.optimize import Bounds

iterNb = 1

def golinski(x):
    return 0.7854*x[0]*x[1]**2*( 3.3333*x[2]**2 + 14.9334*x[2] - 43.0934) \
        -1.5079*x[0]*(x[5]**2 + x[6]**2) + 7.477*(x[5]**3 + x[6]**3) + 0.7854*(x[3]*x[5]**2 + x[4]*x[6]**2)

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
   
def callbackGolinski(Xi):
    global iterNb
    print('{0:4d}   {1: 3.4f}   {2: 3.4f}   {3: 3.4f}   {4: 3.4f}   {5: 3.4f}   {6: 3.4f}   {7: 3.4f}   {8: 3.4f}'\
          .format(iterNb, Xi[0], Xi[1], Xi[2], Xi[3], Xi[4], Xi[5], Xi[6], golinski(Xi)))
    iterNb += 1
    
def genRandNumArray(start, end, num):
    randArray = np.array([])
    
    for j in range(num): 
        res = []
        for idx, val in enumerate(start):
            r = round(random.uniform(start[idx], end[idx]),3)
            res.append(r)
   
        randArray = np.append(randArray, res)
    randArray.shape = (-1, 7)
   
    return randArray
    
def main():
    lowerBounds = [2.6,0.7,17,7.3,7.3,2.9,5.0]
    upperBounds = [3.6,0.8,28,8.3,8.3,3.9,5.5]
    
    noOfRuns = 20 #for genRandNumArray method
    
    randArray = genRandNumArray(lowerBounds, upperBounds, noOfRuns)                       
    
    bounds = Bounds(lowerBounds, upperBounds)
    
    # minimize function
    currRunNb = 1 
    t1_start = perf_counter()
    
    print("randArray is:", randArray)
    
    for a in randArray:
        nonlinear_constraints = NonlinearConstraint(constraints, -np.inf, 0)
        
        print("Run",currRunNb,"initial starting points:" + str(a))
        print('{0:4s}   {1:7s}   {2:7s}   {3:7s}   {4:7s}   {5:7s}   {6:7s}   {7:7s}   {8:7s}'\
              .format('Iter', ' X1', ' X2',  ' X3', ' X4', ' X5', ' X6', ' X7','f(X)'))  
        
       
        
        res = minimize(golinski, a, method="SLSQP", callback=callbackGolinski, jac="2-point", 
                        constraints=[nonlinear_constraints],
                        options={'ftol': 1e-04, 'eps': 0.00001, 'disp': True}, 
                        bounds=bounds)
        
        
        
        print("Run", currRunNb, "Solution: ", res.x, "\n")
        currRunNb += 1

        global iterNb
        iterNb = 1
        
    t1_stop = perf_counter()
    print("Elapsed time:", t1_stop - t1_start) #In fractional seconds
       
if __name__ == '__main__':
    main()