# Feature selection with the Information Gain measure

from numpy import *
from math import log

def infogain(x, y):
    '''
        x: features (data)
        y: output (classes)
    '''
    rows = x.shape[0]
    dim = x.shape[1]

    info_gains = zeros(dim) # features of x
    
    # calculate entropy of the data *hy* with regards to class y
    cl = unique(y)
    y_len = len(y)
    hy = 0
    for i in range(len(cl)):
        c = cl[i]
        py = float(sum(y==c))/len(y) # probability of the class c in the data
        hy = hy+py*log(py,2)
    
    hy = -hy

    # ====================== YOUR CODE HERE ================================
    # Instructions: calculate the information gain for each column (feature)
    
    # O_vc[<feature number (column number)>][<feature value>][<y value>] = N·P(Xj =v,Y =c)
    O_vc = [{} for _ in range(dim)]

    for j in range(dim):
        column = x[:,j]
        O_vc[j] = dict.fromkeys(unique(column)) # For each feature, creates a dict with each unique feature value

        for i in range(rows):
            O_vc[j][column[i]] = dict.fromkeys(cl, 0) # For each feature value, creates a dict with unique y values (so as to sum them and get the total number of <y> encountered with this specific feature value after). Initializes with 0s.

    # Fill O_vc with the good values
    for j in range(0,dim): # For each feature
        for i in range(0,rows): # For each feature_value
            O_vc[j][x.item(i,j)][y.item(i,0)] += 1/y_len

    # Compute info_gains
    for j in range(dim):
        column = x[:,j]
        for v in unique(column):
            Su = sum(column==v) # P(Xj=v)*rows
            hu = 0
            for c in cl:
                pu = 0
                if O_vc[j][v][c] != 0:
                    hu -= O_vc[j][v][c]*log(O_vc[j][v][c],2)

                pu = float(pu)/rows
                if pu != 0:
                    hu -= pu*log(pu,2)

            info_gains[j] -= Su/rows * hu
        
    return info_gains
    
