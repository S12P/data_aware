import numpy as np

#ligne film
#colonne user

def SVD(m):
    return np.linalg.svd(m, full_matrices=False)

def RMSE(M, SVD_sigma, SVD_u, SVD_v):
    res = 0
    for i in range(len(M)):
        for j in range(len(M[0])):
            err = M[i][j]
            for k in range(len(SVD_sigma)):
                err -= SVD_sigma[k] * SVD_u[k][i] * SVD_v[k][j] #checker si cest bon
            res += err**2
    return res
    

def init_mean(m):
    for j in range(len(m[0])):
        mean = 0
        count = 0
        for i in range(len(m)):
            if m[i][j] != 0:
                count += 1
                mean += m[i][j]
        mean /= count
        for i in range(len(m)):
            if m[i][j] == 0:
                m[i][j] = mean
        
        
