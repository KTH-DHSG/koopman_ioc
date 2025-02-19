import numpy as np


## ---------------------------- INVERSE LQR ---------------------------- 
# Solve minimization problem (12-14) from https://arxiv.org/pdf/1810.12590.pdf


def iLQR(A, B, X_star, U_star):

    M  = X_star.shape[0]
    N = X_star.shape[1]
    n = X_star.shape[2]
    m = U_star.shape[2]
    print("First dimension check : " + str(M*(N-2)*m - n*(n+1)/2) + " >= 0")

    AA = compute_AA(A, B, X_star)

    vecQ = np.linalg.lstsq(AA, -U_star[:,0:N-2,:].reshape(1,-1).T, rcond=None)[0]
    Q_re = vecQ.reshape(X_star.shape[2], -1, order='F').T

    return Q_re


def compute_AA(A, B, X_star):

    N = X_star.shape[1]

    AA = 0

    # Compute "controllability matrix"
    # Save it backwards
    BB = np.vstack(
        [B.T @ np.linalg.matrix_power(A.T, i)
                for i in range(N-3, -1, -1)])

    for i in range(1,N-1):
        xx = X_star[:,N-1-i,:] #np.array(x_star[N-1-i,:].T)

        AA += np.kron(xx,BB)
        BB[:-B.T.shape[0],:] = BB[B.T.shape[0]:,:]
        BB[-B.T.shape[0]:,:] *= 0

        """
        xx = np.array(X_star[:,i,:])

        # Select a snippet of it for each i
        BB_cut = BB[(BB.shape[0]-1*i):,:]
        BB_i = np.vstack([BB_cut] + [np.zeros((BB.shape[0]-1*i,BB.shape[1]))])

        AA += np.kron(xx,BB_i)
        """

    print("Rank of AA : " + str(np.linalg.matrix_rank(AA)))
    print("Dims of AA : " + str(AA.shape))
    
    return AA