import numpy as np

def distance(a,b):
    return pow(np.linalg.norm(a-b),2)

def gaussianKernel(X1, X2, sigma = 0.1):
    m = X1.shape[0]
    n = X2.shape[0]
    K = np.zeros((m,n))
    
    # ====================== YOUR CODE HERE =======================
    # Instructions: Calculate the Gaussian kernel (see the assignment
    #				for more details).
    
    den = (2*pow(sigma,2))

    for i in range(m):
        for j in range(n):
            K[i][j] = np.exp(distance(X1[i],X2[j])/den)

    # =============================================================

    return K