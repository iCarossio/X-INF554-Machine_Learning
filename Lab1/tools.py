from numpy import *

def poly_exp(X, degree):
    N,D = X.shape
    for d in range(2,degree+1):
        X = column_stack([X,X[:,0:D]**d])
    return X

def MSE(yt,yp):
    N = len(yt)
    return 1/N*sum((yt-yp)**2)