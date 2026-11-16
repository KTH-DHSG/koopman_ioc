import numpy as np
from tqdm import tqdm
from scipy.optimize import nnls


## ------------------------- INVERSE Bilinear PMP -------------------------
# Solve bilinear PMP version of the minimization problem (12-14) from 
# https://arxiv.org/pdf/1810.12590.pdf


def iBiPMP(A, B, X_star, U_star):

    #print(X_star)
    #print(U_star)

    M  = X_star.shape[0]
    N = X_star.shape[1]
    n = X_star.shape[2]
    m = U_star.shape[2]
    print("First dimension check : " + str(M*(N-2)*m - n*(n+1)/2) + " >= 0")

    AAB = []
    for j in tqdm(range(M)):

        ABj_T = []
        OBj_T = []
        for i in range(N-1):
            AB_aux = A + B[:,:n] * U_star[j,i,0] + B[:,n:] * U_star[j,i,1]
            ABj_T.append(AB_aux.T)
            OB_aux = np.hstack((B[:,:n] @ X_star[j,i,:].reshape(n,1), B[:,n:] @ X_star[j,i,:].reshape(n,1)))
            OBj_T.append(OB_aux.T)
        ABj_T = np.array(ABj_T)
        OBj_T = np.array(OBj_T)
        #print(ABj_T)
        #print(OBj_T)

        diag_OBj_T = np.zeros((m*(N-2),n*(N-2)))
        for i in range(N-2):
            diag_OBj_T[i*m:(i+1)*m,i*n:(i+1)*n] = OBj_T[i,:,:]
        #print(diag_OBj_T)

        AABj = 0
        long_ABj_T = np.zeros((n*(N-2),n))
        for i in range(1,N-1):
            # Multiply all elements by ABj_T element i-1
            for z in range(N):
                long_ABj_T[z*n:(z+1)*n,:] = long_ABj_T[z*n:(z+1)*n,:] @ ABj_T[i-1,:,:]
            
            # Set next new nonzero m rows to identity
            long_ABj_T[(i-1)*n:i*n:] = np.identity(n)
            
            BBj = diag_OBj_T @ long_ABj_T
            #print(BBj)

            xx = X_star[j,i,:].T
            AABj += np.kron(xx,BBj)
        AAB.append(AABj)
        #print(long_ABj_T)
        #print(BBj)

    AAB = np.array(AAB).reshape(M*(N-2)*m,n*n)

    print("Rank of AAB : " + str(np.linalg.matrix_rank(AAB)))
    print("Dims of AAB : " + str(AAB.shape))


    # # If we take the full matrix AAB -> no symmetric constraints enforced
    # vecQ = np.linalg.lstsq(AAB, -U_star[:,0:N-2,:].reshape(1,-1).T, rcond=None)[0]
    # Q_re = vecQ.reshape(X_star.shape[2], -1, order='F').T

    # # If we only consider the upper triangular elements of Q -> symmetric constraints enforced
    # row_indices, col_indices = np.triu_indices(X_star.shape[2])
    # upper_tri_indices = row_indices * X_star.shape[2] + col_indices
    # vecQ = np.linalg.lstsq(AAB[:, upper_tri_indices], -U_star[:,0:N-2,:].reshape(1,-1).T, rcond=None)[0]
    # Q_re = np.zeros((X_star.shape[2], X_star.shape[2]))
    # Q_re[row_indices, col_indices] = vecQ.ravel()
    # Q_re = Q_re + Q_re.T - np.diag(Q_re.diagonal())
    # print("Rank of constrained AAB : " + str(np.linalg.matrix_rank(AAB[:, upper_tri_indices])))
    # print("Dims of constrained AAB : " + str(AAB[:, upper_tri_indices].shape))

    # Diagonal positive-semidefinite cost estimation
    diagonal_indices = [i * (n + 1) for i in range(n)]

    G = AAB[:, diagonal_indices]
    b = -U_star[:, :N-2, :].reshape(-1)

    # Exclude numerically unidentifiable constant-cost features
    norms = np.linalg.norm(G, axis=0)
    active = norms > 1e-9 * norms.max()

    vecQ = np.zeros((n, 1))
    vecQ[active, 0] = nnls(G[:, active], b)[0]

    Q_re = np.diag(vecQ.ravel())

    assert np.min(np.linalg.eigvalsh(Q_re)) >= -1e-10
    print("Rank of constrained AAB : " + str(np.linalg.matrix_rank(AAB[:, diagonal_indices])))
    print("Dims of constrained AAB : " + str(AAB[:, diagonal_indices].shape))


    print("Computed Q: ")
    print(vecQ)

    return Q_re