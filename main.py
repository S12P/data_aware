import numpy as np
import test
import matplotlib.pyplot as plt


def SVD_(m):
    """
        SVD function
    """
    return np.linalg.svd(m, full_matrices=False)

def rank_k(U, S, V, K):
    """
        function which return U, S and V with K-rank
    """
    U_ = np.copy(U[:K])
    S_ = np.copy(S[:K])
    V_ = np.copy(V[:K])
    return (U_, S_, V_)

def SVD_k(m, K):
    """
        function wich return U, S, V from SVD algorithm with K-rank
    """
    U, S_, V = SVD_(m)

    U, S_, V = rank_k(U.T, S_, V, K)

    U = U.T

    S = np.array([[0. for j in range(K)] for i in range(K)])

    for k in range(K):
        S[k][k] = S_[k]
    return U, S, V

def RMSE_original(M, P, R):
    """
        Input:
        - M actual rating
        - P prediction
    """

    result = 0
    for i,j,_ in R:
        result += (M[i][j] - P[i][j])**2
    return result


def RMSE(M, K, U, S, V, R):
    """
        RMSE training

        Input:
            - U, S, V from SVD
            - M original matrix
            - K K-rank
            - R the positions of values in M
    """
    res = 0
    I, J = np.shape(M)
    for i, j in R:
        err = M[i][j]
        for k in range(K):
            err -= S[k][k] * U[i][k] * V[k][j]
        res += err**2
    return res


def EM(M, K, TS, DV, Ori):
    """
        M matrix
        K k-rate
        TS training set
        DV delete values
        Ori original matrix

        Here we use expectation maximisation and we initialize values to 0

    """
    I, J = np.shape(M)
    P = np.array([[0. for j in range(J)] for i in range(I)])

    x = []
    y = []
    z = []

    for step in range(60):
        U, S, V = SVD_k(P, K)
        SVD = np.dot(np.dot(U, S), V)

        for i in range(I):
            for j in range(J):
                if (i, j) in TS:
                    P[i][j] = M[i][j]
                else:

                    P[i][j] = SVD[i][j]

        x += [step]
        y += [RMSE(M, K, U, S, V, TS)]
        z += [RMSE_original(Ori, P, DV)]

    fig, ax = plt.subplots(1, figsize=(8, 6))

    # Set the title for the figure
    fig.suptitle('EM ', fontsize=15)

    # Draw all the lines in the same plot, assigning a label for each one to be
    # shown in the legend
    ax.plot(x, y, color="red", label="RMSE")
    ax.plot(x, z, color="green", label="Test RMSE")
    ax.legend(loc="upper right", title="", frameon=False)
    return P, y, z

def mean_user(M):
    """
        return an array with the mean of each user
    """
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

def similarity(M, mean):
    """
        We use the similarity algorithm from the article Item-Based Collaborative Filtering Recommendation Algorithms (http://files.grouplens.org/papers/www10_sarwar.pdf)
    """
    I, J = np.shape(M)
    sim = np.array([[0. for j in range(I)] for i in range(I)])
    for i in range(J):
        for j in range(J):
            array_user = []
            for user in range(I):
                if M[user][i] != 0 and M[user][j] != 0:
                    array_user += [user]
            a = 0
            b = 0
            c = 0
            for k in range(len(array_user)): # a/(b*c)
                u = array_user[k]
                a += (M[u][i] - mean[u])*(M[u][j] - mean[u])
                b += (M[u][i] - mean[u])**2
                c += (M[u][j] - mean[u])**2
            if len(array_user) != 0 and b != 0 and c != 0:
                sim[i][j] = a / (np.sqrt(b) * np.sqrt(c))
    return sim


def IIS(M, R): #item-item similarity based recommender
    """
        This algortihm init none values in the matrix
    """
    I, J = np.shape(M)
    mean = mean_user(M)
    S = similarity(M, mean)
    F = np.array([[0. for j in range(J)] for i in range(I)])
    W = np.array([[0. for j in range(J)] for i in range(I)])
    H = np.array([[mean[i] for j in range(J)] for i in range(I)])
    for i,j in R:
        F[i][j] = (M[i][j] - mean[i]) / 100.
    FS = np.dot(F, S)
    for i in range(I):
        for j in range(J):
            if (i, j) not in R:
                W[i][j] = H[i][j] + FS[i][j]
    return W


def EM2(M, K, TS, DV, Ori):
    """
    M matrix
    K k-rate
    TS training set
    DV delete values
    Ori original matrix

    Here we use expectation maximisation and we initialize values with item-item similarity based recommender
    """
    I, J = np.shape(M)
    P = IIS(M.T, R).T #because our function ISS use the tranpose

    x = []
    y = []
    z = []

    for step in range(60):
        U, S, V = SVD_k(P, K)
        SVD = np.dot(np.dot(U, S), V)

        for i in range(I):
            for j in range(J):
                if (i, j) in TS:
                    P[i][j] = M[i][j]
                else:

                    P[i][j] = SVD[i][j]

        x += [step]
        y += [RMSE(M, K, U, S, V, TS)]
        z += [RMSE_original(Ori, P, DV)]
    fig, ax = plt.subplots(1, figsize=(8, 6))

    # Set the title for the figure
    fig.suptitle('EM2 ', fontsize=15)

    # Draw all the lines in the same plot, assigning a label for each one to be
    # shown in the legend
    ax.plot(x, y, color="red", label="RMSE")
    ax.plot(x, z, color="green", label="Test RMSE")
    ax.legend(loc="upper right", title="", frameon=False)
    return P, y, z


K = 20
M, test_values, R = test.train_matrix(.30)
M = np.array(M)
M = M.astype(float)
Ori, _, _ = test.train_matrix(0)
Ori = np.array(Ori)
Ori = Ori.astype(float)

W1, x1, y1 = EM(M, K, R, test_values, Ori)

W2, x2, y2 = EM2(M, K, R, test_values, Ori)

x = [k for k in range(60)]

fig, ax = plt.subplots(1, figsize=(8, 6))

# Set the title for the figure
fig.suptitle('EM && EM2 ', fontsize=15)

# Draw all the lines in the same plot, assigning a label for each one to be
# shown in the legend
ax.plot(x, x1, color="red", label="EM RMSE")
ax.plot(x, y1, color="green", label="EM Test RMSE")

ax.plot(x, x2, color="blue", label="EM2 RMSE")
ax.plot(x, y2, color="yellow", label="EM2 Test RMSE")
ax.legend(loc="upper right", title="", frameon=False)
plt.show()
