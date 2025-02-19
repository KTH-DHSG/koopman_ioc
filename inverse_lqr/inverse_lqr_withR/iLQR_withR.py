import numpy as np
from scipy.linalg import null_space


## ---------------------------- INVERSE LQR ---------------------------- 
# Solve minimization problem (12-14) from https://arxiv.org/pdf/1810.12590.pdf


def iLQR_withR(A, B, X_star, U_star):

    M  = X_star.shape[0]
    N = X_star.shape[1]
    n = X_star.shape[2]
    m = U_star.shape[2]
    print("First dimension check : " + str(M*(N-2)*m - n*(n+1)/2) + " >= 0")
    print("Second dimension check : " + str(M*(N-2)*m - n*n+m*m) + " >= 0")

    AA = compute_AA(A, B, X_star)

    uu = np.kron(np.identity(m),U_star[:,0:N-2,:].reshape(m,-1).T)

    AAuu = np.concatenate((AA, uu), axis=1)

    print("Rank of AAuu : " + str(np.linalg.matrix_rank(AAuu)))
    print("Dims of AAuu : " + str(AAuu.shape))

    sol = null_space(AAuu)

    """
    AAuu_T_AAuu = np.dot(AAuu.T, AAuu)
    eigen_values, eigen_vectors = np.linalg.eig(AAuu_T_AAuu)
    eigen_vector_min = eigen_vectors[:, np.argmin(eigen_values)]
    sol = eigen_vector_min
    """

    vecQ = sol[0:n*n]
    Q_re = vecQ.reshape(X_star.shape[2], -1, order='F').T

    vecR = sol[n*n:]
    R_re = vecR.reshape(U_star.shape[2], -1, order='F').T

    return Q_re, R_re


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