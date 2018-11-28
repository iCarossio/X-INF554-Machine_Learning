import itertools
import numpy as np
from functools import reduce

def mapFeatures(X, degree=6):
    '''
    Generate a new feature matrix consisting of all polynomial combinations of 
    the features with degree less than or equal to the specified degree. 
    '''

    # TODO
    F = list()
    for x in X:
        
        # Generate all possible cobinations of factors from degree 1 to degree
        polys = []
        for d in range(1,degree):
            polys.extend(itertools.combinations_with_replacement(x, r=d))
            
        # Multiply list of factors
        prods = [reduce(lambda x, y: x*y, factors) for factors in polys]
        #prods = [factors for factors in polys]
        F.append([1]+prods)

    return np.array(F)