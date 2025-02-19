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

    """
    if np.linalg.matrix_rank(AA) < AA.shape[1]:
        X_star = X_star[:,:-int(N/6),:]
        print("New Dims of X* : " + str(X_star.shape))
        U_star = U_star[:,:-int(N/6),:]
        print("New Dims of U* : " + str(U_star.shape))
        N = X_star.shape[1]

        AA = compute_AA(A, B, X_star)
    """

    # If we take the full matrix AAA -> no symmetric constraints enforced
    vecQ = np.linalg.lstsq(AA, -U_star[:,0:N-2,:].reshape(1,-1).T, rcond=None)[0]
    Q_re = vecQ.reshape(X_star.shape[2], -1, order='F').T

    # # If we only consider the upper triangular elements of Q -> symmetric constraints enforced
    # row_indices, col_indices = np.triu_indices(X_star.shape[2])
    # upper_tri_indices = row_indices * X_star.shape[2] + col_indices
    # vecQ = np.linalg.lstsq(AA[:, upper_tri_indices], -U_star[:,0:N-2,:].reshape(1,-1).T, rcond=None)[0]
    # Q_re = np.zeros((X_star.shape[2], X_star.shape[2]))
    # Q_re[row_indices, col_indices] = vecQ.ravel()
    # Q_re = Q_re + Q_re.T - np.diag(Q_re.diagonal())

    # # If we only consider the diagonal elements of Q -> symmetric & zero off-diagonal constraints enforced
    # diagonal_indices = [i * (X_star.shape[2] + 1) for i in range(X_star.shape[2])]
    # vecQ = np.linalg.lstsq(AA[:, diagonal_indices], -U_star[:,0:N-2,:].reshape(1,-1).T, rcond=None)[0]
    # Q_re = np.zeros((X_star.shape[2], X_star.shape[2]))
    # np.fill_diagonal(Q_re, vecQ)


    print("Vectorized Q: ")
    print(vecQ)

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