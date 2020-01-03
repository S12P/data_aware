import numpy as np
import test


def SVD_(m):
    return np.linalg.svd(m, full_matrices=False)

def rank_k(U, S, V, K):
    U_ = np.copy(U[:K])
    S_ = np.copy(S[:K])
    V_ = np.copy(V[:K])
    return (U_, S_, V_)

def SVD_k(m, K):
    U, S_, V = SVD_(m)
    
    U, S_, V = rank_k(U.T, S_, V, K)
    
    U = U.T
    
    S = np.array([[0. for j in range(K)] for i in range(K)])
    
    for k in range(K):
        S[k][k] = S_[k]
    return U, S, V

def RMSE_original(O, P, R):
    """
        Input:
        - O actual rating
        - P prediction
        """
    assert(len(O) == len(P))
    assert(len(O[0]) == len(P[0]))
    
    result = 0
    for i,j in R:
        result += (O[i][j] - P[i][j])**2
    return result


def RMSE(M, K, U, S, V, R):
    res = 0
    I, J = np.shape(M)
    for i, j in R:
        err = M[i][j]
        for k in range(K):
            err -= S[k][k] * U[i][k] * V[k][j] #checker si cest bon pr indice de U
        res += err**2
    return res


def EM(O, K, R):
    """
        O original matrix
        K k-rate
        R training set
        
        """
    I, J = np.shape(O)
    P = np.array([[0. for j in range(J)] for i in range(I)])
    
    U, S, V = SVD_k(O, K)
    SVD = np.dot(np.dot(U, S), V)
    for i in range(I):
        for j in range(J):
            if (i, j) in R:
                P[i][j] = O[i][j]
            else:
                P[i][j] = SVD[i][j]
                    
    for step in range(100):
        U, S, V = SVD_k(P, K)
        SVD = np.dot(np.dot(U, S), V)

        for i in range(I):
            for j in range(J):
                if (i, j) in R:
                    P[i][j] = O[i][j]
                else:
                    
                    P[i][j] = SVD[i][j]
                        
        print(RMSE(M, K, U, S, V, R))
    return P
                    
####################################################

# ligne user
# colonne film

def mean_user(M):
    I, J = np.shape(M)
    m = np.zeros(I) #array of mean
    for i in range(I):
        mean = 0
        count = 0
        for j in range(J):
            if M[i][j] != 0:
                count += 1
                mean += M[i][j]
        mean /= count
        m[i] = mean
    return m

def similarity(MM):
    M = np.copy(MM.T) # pour avoir les films en ligne
    I, J = np.shape(M)
    sim = np.array([[0 for j in range(I)] for i in range(I)])
    for i in range(I):
        for j in range(I):
            if i != j:
                nI = np.sqrt(np.dot(M[i], M[i]))
                nJ = np.sqrt(np.dot(M[j], M[j]))
                sim[i][i] = (np.dot(M[i], M[j])) / (nI * nJ)
            else:
                sim[i][i] = 1
    return sim


def IIS(M, R): #item-item similarity based recommender
    I, J = np.shape(M)
    mean = mean_user(M)
    S = similarity(M)
    F = np.array([[0 for j in range(J)] for i in range(I)])
    W = np.array([[0 for j in range(J)] for i in range(I)])
    H = np.array([[mean[i] for j in range(J)] for i in range(I)])
    for i in range(I):
        for j in range(J):
            if (i, j) in R:
                F[i][j] = (M[i][j] - mean[i]) / 100.
    for i in range(I):
        for j in range(J):
            if (i, j) in R:
                W[i][j] = H[i][j] + np.dot(F, S)[i][j]
    return W


def EM2(O, K, R):
    """
    O original matrix
    K k-rate
    R training set
    
    """
    I, J = np.shape(O)
    P = IIS(O.T, R).T
    
    U, S, V = SVD_k(O, K)
    SVD = np.dot(np.dot(U, S), V)
    for i in range(I):
        for j in range(J):
            if (i, j) in R:
                P[i][j] = O[i][j]
            else:
                P[i][j] = SVD[i][j]
    
    for step in range(100):
        U, S, V = SVD_k(P, K)
        SVD = np.dot(np.dot(U, S), V)
        
        for i in range(I):
            for j in range(J):
                if (i, j) in R:
                    P[i][j] = O[i][j]
                else:
                    
                    P[i][j] = SVD[i][j]
        
        print(RMSE(M, K, U, S, V, R))
    return P



#M, test_values, R = test.train_matrix(.10)
#M = np.array(M)
#M = M.astype(float)
#Ori, _, _ = test.train_matrix(0)
#Ori = np.array(Ori)
#Ori = Ori.astype(float)
#W = EM(M, 10, R)
#print(RMSE_original(Ori, W))

M, test_values, R = test.train_matrix(.10)
M = np.array(M).T #transposé ici
M = M.astype(float)
Ori, _, _ = test.train_matrix(0)
Ori = np.array(Ori)
Ori = Ori.astype(float)
W = EM2(M, 10, R)
print(RMSE_original(Ori, W, R))
