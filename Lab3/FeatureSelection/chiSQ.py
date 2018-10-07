# Feature selection with the Chi^2 measure

from collections import Counter
from numpy import *

def chiSQ(x, y):
    '''
        x: features (data)
        y: output (classes)
    '''
    cl     = unique(y) # unique number of classes
    rows   = x.shape[0]
    dim    = x.shape[1] # dim = N
    chisq  = zeros(dim) # initialize array (vector) for the chi^2 values
    cl_len = len(cl)

    # Y_probs[c] = P(Y=c)*rows
    y_list  = [yj[0] for yj in y.tolist()]
    Y_probs = dict(Counter(y_list)) 

    # X_probs[<feature number j (column number)>][v] = P(Xj=v)*rows
    X_probs = [{} for _ in range(dim)]

    # O_vc[<feature number (column number)>][<feature value>][<y value>] = N·P(Xj =v,Y =c)
    O_vc = [{} for _ in range(dim)]

    for j in range(dim):
        column = x[:,j]
        O_vc[j] = dict.fromkeys(unique(column)) # For each feature, creates a dict with each unique feature value
        X_probs[j] = dict(Counter(column.tolist())) 

        for i in range(rows):
            O_vc[j][column[i]] = dict.fromkeys(cl, 0) # For each feature value, creates a dict with unique y values (so as to sum them and get the total number of <y> encountered with this specific feature value after). Initializes with 0s.

    # Fill O_vc with the good values
    for j in range(0,dim): # For each feature
        for i in range(0,rows): # For each feature_value
            O_vc[j][x.item(i,j)][y.item(i,0)] += 1/cl_len*dim # P(Xj=v,Y=c)·N

    # Compute chisq
    for j in range(dim):
        column = x[:,j]
        for v in unique(column):
            for c in cl:
                E_vc = dim*X_probs[j][v]/rows*Y_probs[c]/rows
                chisq[j] += pow(O_vc[j][v][c] - E_vc,2)/E_vc
  
    return chisq