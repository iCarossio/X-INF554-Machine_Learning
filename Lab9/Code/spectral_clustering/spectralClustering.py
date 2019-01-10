from numpy import *
from scipy.cluster.vq import kmeans2

def spectralClustering(W, k):

    # ====================== ADD YOUR CODE HERE ======================
    # Instructions: Perform spectral clustering to partition the 
    #               data into k clusters. Implement the steps that
    #               are described in Algorithm 2 on the assignment.    
    
    
    D = diag(sum(W,axis=0))
    L = D - W
    
    # eigh for symetric matrix
    eigenValues, eigenVectors = linalg.eigh(L)
    
    # Eigenvectors that correspond to k smallest eigenvalues.
    idx = eigenValues.argsort() 
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    U = eigenVectors[:,:k]

    centroids, labels = kmeans2(U,k,minit='points')

    # =============================================================

    return labels
