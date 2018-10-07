from numpy import *

#data : the data matrix
#k the number of component to return
#return the new data and  the variance that was maintained AND the principal components (ALL)
def pca(data,k):
    # Performs principal components analysis (PCA) on the n-by-p data matrix A (data)
    # Rows of A correspond to observations (wines), columns to variables.
    ## TODO: Implement PCA

    # compute the mean
    # subtract the mean (along columns)
    # compute covariance matrix
    # compute eigenvalues and eigenvectors of covariance matrix
    # Sort eigenvalues
    # Sort eigenvectors according to eigenvalues
    # Project the data to the new space (k-D)

    A         = data
    M         = mean(A, axis=0) # Mean by column
    C         = A-M
    W         = transpose(C)@C
    w,v       = linalg.eig(W) # Valeurs prores w[i], vecteurs propres v[:,i]
    top_k_idx = argsort(w)[-k:] # Get index of the top k eigenvalues
    
    N  = A.shape[1] # dim (number of features)
    Uk = zeros(shape=(N,k))
    eigval_sort = sort(w)
    
    i=0
    for idx in top_k_idx:
        Uk[:,i] = v[:,idx] # extract the kth eigenvectors associated to the k greater eigenvalues
        i+=1

    rep = C@Uk
    var = sum(eigval_sort[:k])/sum(eigval_sort)

    return rep, var, Uk
