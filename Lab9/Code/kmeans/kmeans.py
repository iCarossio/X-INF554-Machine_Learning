from numpy import *
from euclideanDistance import euclideanDistance
from simpleInitialization import simpleInitialization
import copy



def kmeans(X, k):
    # Intialize centroids
    C = simpleInitialization(X, k)
    
    print(C.shape)
    print(X.shape)

    # Initialize var iables
    n = X.shape[0]     # Size of the sample
    m = X.shape[1]     # Space dimension

    iterations = 0
    labels  = zeros(n) 
    oldC    = zeros((k,m))
    C_sizes = zeros(k)
    threshold = 0.1
    
    # ====================== ADD YOUR CODE HERE ======================
    # Instructions: Run the main k-means algorithm. Follow the steps 
    #               given in the description. Compute the distance 
    #               between each instance and each centroid. Assign 
    #               the instance to the cluster described by the closest
    #               centroid. Repeat the above steps until the centroids
    #               stop moving or reached a certain number of iterations
    #               (e.g., 100).
    
    while ((C-oldC)>threshold).any(): # While C is not converging (according to a certain threshold)
        for i in range(n):

            distances = zeros(k)

            for j in range(k):
                distances[j] = euclideanDistance(X[i,:], C[j])

            c_index = argmax(distances)
            labels[i] = c_index
            C_sizes[c_index] += 1
        
        oldC = copy.deepcopy(C)
        
        for j in range(k):
            if C_sizes[j] > 0:
                u = zeros(m)
                Xs_in_Cj = argwhere(labels==j)
                for i in Xs_in_Cj:
                    u = add(u,X[i,:])
                u /= C_sizes[j]
                C[j] = u

        iterations += 1

    # ===============================================================
    print("k-means exectued in {} iterations".format(iterations))
    return labels