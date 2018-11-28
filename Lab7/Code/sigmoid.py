from numpy import *

def proba(theta,x):
    return sigmoid(dot(theta,x))

def sigmoid(z):
    # Computes the sigmoid of z.

    # ====================== YOUR CODE HERE ======================
    # Instructions: Implement the sigmoid function as given in the
    # assignment (and use it to *replace* the line above).
    
    return 1/(1+exp(-z))

