from numpy import *
from time import time
from sys import stdout

def compute_svd(V, r):
    k = r # nombres de valeurs propres significatives conservées
    A = V
    AtA = A@transpose(A)
    tAA = transpose(A)@A

    SpU, U = linalg.eig(AtA) # Valeurs prores w[i], vecteurs propres v[:,i]
    SpV, V = linalg.eig(tAA) # Valeurs prores w[i], vecteurs propres v[:,i]

    S = sqrt(SpU)

    N  = A.shape[1] # dim (number of features)
    M  = A.shape[0] # rows (number of features)

    S_arg_sorted = argsort(S) # Get indexes of the top singular values
    S_sorted     = sort(S)
    X = {}

    S_diag   = diag(S_sorted[-k:]) # Matrice diagonale de valeurs singulières triées
    top_k_idx = S_arg_sorted[-k:] 
    Uk = zeros(shape=(M,k))
    Vk = zeros(shape=(N,k))

    i=0
    for idx in top_k_idx:
        Uk[:,i] = U[:,idx] # extract the kth eigenvectors associated to the k greater singular values
        Vk[:,i] = V[:,idx] 
        i+=1

    U = abs(Uk)
    V = transpose(abs(Vk))
    
    return U, V

def compute_error(V,W,H):
    return pow(linalg.norm(V-W@H),2)

def nmf_factor(V, r, iterations=100):
    n, m = V.shape
    W = ones((n, r))
    H = ones((r, m))
    d_iter = ones(100)

    W, H = compute_svd(V,r)
    
    print(r) # 25
    print(V.shape) # (3012, 1000)
    print(H.shape) # (25, 1000)
    print(W.shape) # (3012, 25)

    for k in range(iterations):
        print("Iteration {}".format(k))
        Wt = transpose(W)
        num = Wt@V
        den = Wt@W@H
        for i in range(r):
            for j in range(m):
                H[i,j] = H[i,j]*num[i,j]/den[i,j]

        Ht = transpose(H)
        num = V@Ht
        den = W@H@Ht
        for i in range(n):
            for j in range(r):
                W[i,j] = W[i,j]*num[i,j]/den[i,j]
        
        d_iter[k] = compute_error(V,W,H)

    return W, H, d_iter
