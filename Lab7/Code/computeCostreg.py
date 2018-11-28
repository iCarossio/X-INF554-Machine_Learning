from numpy import *
from sigmoid import proba

def computeCostreg(theta, X, y, l):
    # Computes the cost of using theta as the parameter for regularized logistic regression.

    m = X.shape[0] # number of training examples
    J = 0

    # ====================== YOUR CODE HERE ======================
    # Instructions: Calculate the error J of the decision boundary
    #               that is described by theta  (see the assignment 
    #               for more details).

    n  = X.shape[1]

    for i in range(m):
        J += y[i]*log(proba(theta,X[i])) + (1-y[i])*log(1-proba(theta,X[i]))
    J *= -1/m

    for j in range(n):
        J += l/(2*m)*theta[j]**2

    return J
