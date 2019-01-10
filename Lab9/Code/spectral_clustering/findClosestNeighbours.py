from numpy import *
from euclideanDistance import euclideanDistance

def findClosestNeighbours(data, k):
    
    n = data.shape[0] # Sample size 

    closestNeighbours = zeros((n, k)).astype(int)
    distances = zeros((n,n))

    # ====================== ADD YOUR CODE HERE ======================
    # Instructions: Find the k closest instances of each instance
    #               using the euclidean distance.
    
    for i in range(n):
        distances[i,i] = 0
        for j in range(i+1,n):
            distances[i,j] = distances[j,i] = euclideanDistance(data[i,:], data[j,:])

        closestNeighbours[i,:] = argsort(distances[i,:])[:k] # sorted(range(len(s)), key=lambda k: s[k])
    # =============================================================

    return closestNeighbours
