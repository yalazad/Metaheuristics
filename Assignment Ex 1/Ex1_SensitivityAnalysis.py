from numpy import *
from scipy.optimize import *
import math

#Fare parameters
a1 = 100
a2 = 150
a3 = 300

#Initialise global variables/arrays to zero, they will be manipulated later
#The below variables are global so that they can be used by the getRevenue function
p1,p2,p3 = 0,0,0
priceBuckets = [p1, p2, p3]

D1,D2,D3 = 0,0,0
passengerBuckets = [D1, D2, D3]

noOfSeats = 153
    
def myFunction(z):
    dLambda = z
    
    fareParams = [a1,a2,a3]
    
    #Populate the priceBuckets array
    global priceBuckets
        
    for idx, val in enumerate(priceBuckets):
        priceBuckets[idx] = getpVal(fareParams[idx],dLambda)
    
    #Populate the passengerBuckets array
    global passengerBuckets
    
    for idx2, val2 in enumerate(passengerBuckets):
        passengerBuckets[idx2] = getDVal(fareParams[idx2],priceBuckets[idx2])
    
    #F = D1 + D2 + D3 - noOfSeats
    F = passengerBuckets[0] + passengerBuckets[1] + passengerBuckets[2] - noOfSeats
    return F

#Seat demand calculation
def getDVal(ai,pi):
    #D = ai * exp(-1/ai*pi)
    D = ai * math.exp(-1/ai*pi)
    return D

#Price bucket calculation
def getpVal(ai, dLambda):
    #pi = ai + λ
    p = ai + dLambda
    return p

def getRevenue():
    #Revenue function: f(p1,p2,p3) = p1D1 + p2D2 + p3D3
    r = (priceBuckets[0] * passengerBuckets[0]) + (priceBuckets[1] * passengerBuckets[1]) + (priceBuckets[2] * passengerBuckets[2])
    return r
    
zGuess = 2

z = fsolve(myFunction,zGuess)
r = getRevenue()
print("The total number of seats is:  "+ str(noOfSeats))
print("Lambda is:  "+ str(z))
print("Price buckets :- p1 is:" +str(priceBuckets[0]) + ", p2 is:" +str(priceBuckets[1]) + ", p3 is:" +str(priceBuckets[2]))
print("No. of passengers :- D1 is:" +str(round(passengerBuckets[0],2)) + ", D2 is:" +str(round(passengerBuckets[1],2)) + ", D3 is:" +str(round(passengerBuckets[2],2)))
print("Revenue is:  "+ str(r))