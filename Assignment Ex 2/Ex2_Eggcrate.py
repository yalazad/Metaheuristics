import numpy as np
from scipy.optimize import minimize
from time import perf_counter
import random
import math

iter_nb = 1

def eggcrate(x):
    """
        The Eggcrate function.
    """
    #x^2 + y^2 + 25(sin^2(x) + sin^2(y))
    return sum(x[:-1]**2.0 + x[1:]**2.0 + 25 * (math.sin(x[:-1])**2.0 + math.sin(x[1:])**2.0))

def eggcrate_grad(x):
    """
        Gradient of the Eggcrate function.
    """
    #(2x+25sin(2x),2y+25sin(2y))
    x = np.asarray(x)
    der = np.zeros_like(x)
    der[0] = 2*x[0] + 25 * math.sin(2*x[0])
    der[1] = 2*x[1] + 25 * math.sin(2*x[1])
    #print("Eggcrate x is:", x, " gradient is:", der)
    return der

def eggcrate_hess(x):
    """
        Hessian of the Eggcrate function.
    """
    x = np.asarray(x)
    xx = 100*math.cos(x[0])**2-48
    yx = 0
    xy = 0
    yy = 100*math.cos(x[1])**2-48
    H = np.array([[xx,yx],[yx,yy]])
    #print("Eggcrate x is:", x, " Hessian is:",H)  
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

def callbackEggcrate(Xi):
    global iter_nb
    print('{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}'.format(iter_nb, Xi[0], Xi[1], eggcrate(Xi)))
    iter_nb += 1

    
def main():
    # initial starting point
    lower_bound = -2 * math.pi
    upper_bound = 2 * math.pi
    no_of_runs = 20
    randArray = genRandNumArray(lower_bound, upper_bound, no_of_runs)
    print("randArray is:", randArray)
 
    curr_run_nb = 1 
    
    # Monitor performance
    t1_start = perf_counter()
    
    for e in randArray:
        print("Run", curr_run_nb, "- Eggcrate Initial starting points:" + str(e))
        
        # minimize function
        print('{0:4s}   {1:9s}   {2:9s}   {3:9s}'.format('Iter', ' X1', ' X2', 'f(X)'))  
        
        res = minimize(eggcrate, e, method='Newton-CG', callback=callbackEggcrate,
            jac=eggcrate_grad, hess=eggcrate_hess,
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