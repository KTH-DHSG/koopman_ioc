import numpy as np
import casadi as ca
from tqdm import tqdm

import sys


## ------------------------- INVERSE Bilinear PMP -------------------------
# Solve bilinear PMP version of the minimization problem 

def iBiPMP(A, B, X_star, U_star):

    M  = X_star.shape[0]
    N = X_star.shape[1]
    n = X_star.shape[2]
    m = U_star.shape[2]

    # Symbolic state and control variables
    x = ca.MX.sym('x',n)
    u1 = ca.MX.sym('u1')
    u2 = ca.MX.sym('u2')

    # State dynamics
    f = A @ x + B[:,:n] @ x * u1 + B[:,n:] @ x * u2
    f_x = ca.jacobian(f, x)
    f_u = ca.jacobian(f, ca.vertcat(u1,u2))

    P_bar = []
    for j in tqdm(range(M)):
        A_ = A_bar(X_star[j,:,:], U_star[j,:,:], ca.Function('f_x', [x, u1, u2], [f_x]))
        B_ = B_bar(X_star[j,:,:], U_star[j,:,:], ca.Function('f_u', [x, u1, u2], [f_u]))
        Q_ = Q_bar(X_star[j,:,:])
        R_ = R_bar(U_star[j,:,:])
        S_ = S_bar(X_star[j,:,:], U_star[j,:,:], ca.Function('f_x', [x, u1, u2], [f_x]))

        # Calculate F and E matrices
        T = np.concatenate((ca.mtimes(ca.mtimes(B_, ca.inv(A_)), Q_), R_), axis=1)
        F = ca.mtimes(ca.mtimes(B_, ca.inv(A_)), S_)

        # Check for IOC effectiveness
        if np.linalg.matrix_rank(F) == X_star.shape[2]:
            P = T - ca.mtimes(ca.mtimes(F, ca.inv(ca.mtimes(F.T, F))), ca.mtimes(F.T, T))
        else:
            print("Not IOC effective")
            continue # Skip this segment

        P_bar.append(P)

    P_bar = np.vstack(P_bar)

    # Solve centralized, single-agent problem
    w = ca.SX.sym('w', n*n+m*m)
    w_sum = ca.vertcat(w[0])
    lb = ca.vertcat(1)
    ub = ca.vertcat(1)

    # Define the optimization problem
    f_obj = 1/2 * ca.mtimes(w.T, ca.mtimes(ca.mtimes(P_bar.T, P_bar), w))**2
    prob = {'f': f_obj, 'x': w, 'g': w_sum}
    solver_opts = {'ipopt': {'max_iter': 1000}, 'ipopt.print_level':0, 'print_time':0}
    solver = ca.nlpsol('solver', 'ipopt', prob, solver_opts)

    # Solve the optimization problem
    sol = solver(x0=0, lbg=lb, ubg=ub)
    
    vecQ = np.array(sol['x'][0:n*n])
    #print(vecQ)
    Q_re = vecQ.reshape(X_star.shape[2], -1, order='F').T

    vecR = np.array(sol['x'][n*n:])
    #print(vecR)
    R_re = vecR.reshape(U_star.shape[2], -1, order='F').T

    return Q_re, R_re


def A_bar(states, controls, f_x):
    N, n = states.shape[0], states.shape[1]
    ret = np.eye(n * (N-1))

    for i in range(N-2):
        ret[n*i:n*(i+1), n*(i+1):n*(i+2)] = -f_x(states[i+1, :], controls[i+1, 0], controls[i+1, 1]).T
    
    return ret


def B_bar(states, controls, f_u):
    N, n = states.shape[0], states.shape[1]
    m = controls.shape[1]
    ret = np.zeros((m * (N-1), n * (N-1)))

    for i in range(N-1):
        ret[m*i:m*(i+1), n*i:n*(i+1)] = f_u(states[i, :], controls[i, 0], controls[i, 1]).T

    return ret


def Q_bar(states):
    N, n = states.shape[0], states.shape[1]
    ret = []

    for i in range(1,N):
        ret.append(np.kron(states[i, :], np.eye(n)))

    return np.vstack(ret)


def R_bar(controls):
    N, m = controls.shape[0], controls.shape[1]
    ret = []

    for i in range(N-1):
        ret.append(np.kron(controls[i, :], np.eye(m)))

    return np.vstack(ret)


def S_bar(states, controls, f_x):
    N, n = states.shape[0], states.shape[1]
    ret = np.zeros((n * (N-1), n))

    ret[-n:, :] = f_x(states[-1, :], controls[-1, 0], controls[-1, 1]).T

    return ret