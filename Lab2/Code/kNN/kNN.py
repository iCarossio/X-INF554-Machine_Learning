from numpy import *
from numpy.linalg import norm
import operator

def kNN_prediction(k, X, labels, x):
    '''
    kNN classification of x
    -----------------------
        Input: 
        k: number of nearest neighbors
        X: training data           
        labels: class labels of training data
        x: test instance

        return the label to be associated with x

        Hint: you may use the function 'norm' 
    '''

    # Compute the norm for each 
    distances = {}
    i=0
    for training in X:
        distances[i] = norm(training-x)
        i = i+1
    
    # Sorted by distance
    distances = sorted(distances.items(), key=operator.itemgetter(1))[:k]
    
    # classes = { (i, (total_distance, number_of_i) }
    classes = {}
    for i, distance in distances:
        if not labels[i] in classes:
            classes[labels[i]] = (distance, 1)
        else:
            classes[labels[i]] = tuple(map(sum, zip((distance,1), classes[labels[i]])))

    # results = { (i, average_distance) }
    results = {}
    for classe, couple in classes.items():
        results[classe] = couple[0]/couple[1]

    results = sorted(results.items(), key=operator.itemgetter(1))[0][0]

    return results

 
