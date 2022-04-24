import numpy as np
from scipy.optimize import minimize
from time import perf_counter
import random
import math

iter_nb = 1

def rosen(x):
    """
        The Rosenbrock Banana function
    """
    #f(x) =  100*(x_i - x_{i-1}^2)^2 + (1- x_{1-1}^2)
    return 100.0*(x[1]-x[0]**2.0)**2.0 + (1-x[0])**2.0

def rosen_grad(x):
    """
        Gradient of the Rosenbrock function
    """
    x = np.asarray(x)
    grad = np.zeros_like(x)
    grad[0] = -400*x[0]*(x[1] - x[0]**2) -2*(1-x[0])
    grad[1] = 200*(x[1]-x[0]**2)
    #print("x is:", x, " gradient is:", grad)
    return grad

def rosen_hess(x):
    """
        Hessian of the Rosenbrock function
    """
    x = np.asarray(x)
    xx = 1200*x[0]**2-400*x[1]+2
    yx = -400*x[0]
    xy = -400*x[0]
    yy = 200
    H = np.array([[xx,yx],[yx,yy]])
    #print("Rosen x is:", x, " Hessian is:",H)  
    return H
    
def genRandNumArray(lower_bound, upper_bound, no_of_runs):
    coords = list()
    randArray = np.array([]) 
    
    for x in range(no_of_runs):
        r = round(random.uniform(lower_bound, upper_bound), 2)
        # Use the same values for X and Y
        coords.append([r,r])

    randArray = np.append(randArray, np.array(coords))
    randArray.shape = (20,2)

    return randArray

def callbackRosen(Xi):
    global iter_nb
    print('{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}'.format(iter_nb, Xi[0], Xi[1], rosen(Xi)))
    iter_nb += 1

    
def main():
    # initial starting points
    lower_bound = -5
    upper_bound = 5
    no_of_runs = 20
    randArray = genRandNumArray(lower_bound, upper_bound, no_of_runs)
    print("randArray is:", randArray)
 
    curr_run_nb = 1 
    
    # Monitor performance
    t1_start = perf_counter()
        
    for k in randArray:
        print("Run", curr_run_nb, "- Rosenbrock Initial starting points:" + str(k))
        
        # minimize function
        print('{0:4s}   {1:9s}   {2:9s}   {3:9s}'.format('Iter', ' X1', ' X2', 'f(X)'))  
        
        res = minimize(rosen, k, method='Newton-CG', callback=callbackRosen,
            jac=rosen_grad, hess=rosen_hess,
                    options={'xtol': 1e-8, 'disp': True})
        
        print("Run", curr_run_nb, "- Solution: ", res.x, "\n")
        curr_run_nb += 1
        
        #Reset iteration number
        global iter_nb
        iter_nb = 1
        
    t1_stop = perf_counter()
    print("Elapsed time:", t1_stop - t1_start)#In fractional seconds
 
if __name__ == '__main__':
    main() 