from numpy import *
from sigmoid import proba

def computeGradreg(theta, X, y, l):
    # Computes the gradient of the cost with respect to the parameters.

    m = X.shape[0] # number of training examples
    
    grad = zeros_like(theta) #initialize gradient   

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the gradient of cost for each theta,
    # as described in the assignment.

    for i in range(m):
        grad[0] += (proba(theta,X[i])-y[i])*X[i,0]

    for j in range(1, len(grad)):
        for i in range(m):
            grad[j] += (proba(theta,X[i])-y[i])*X[i,j]
        grad[j] += l*theta[j]

    grad *= 1/m

    # =============================================================

    return grad
    
