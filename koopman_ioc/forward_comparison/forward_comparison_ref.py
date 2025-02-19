import numpy as np
import matplotlib.pyplot as plt
import casadi as ca

import sympy as sp
from sympy.polys.orderings import monomial_key
from sympy.polys.monomials import itermonomials


# Supress scientific notation when printing NumPy arrays
np.set_printoptions(precision=5,suppress=True)


def generateTrajectory(start_pt, T, z, phi, f):
    # Declare CasADi decision variables
    u = ca.SX.sym('u', T)

    # Dynamics and control cost
    px, py = start_pt[0], start_pt[1]
    traj_x = []
    traj_y = []
    cost_k = 0

    for k in range(T):
        next = f(px, py, u[k])
        px = next[0]
        py = next[1]
        traj_x.append(px)
        traj_y.append(py)
        phi_k = phi(px, py, u[k])
        cost_k += ca.dot(z, phi_k)

    cost_x_T = 0  # (px - end_pt[0])**2 + (py - end_pt[1])**2
    f_cost = (1 / T) * cost_k + cost_x_T

    # CasADi decision variables, constraints, and objective function
    u_max = np.inf
    traj_x_concat = ca.vertcat(*traj_x)
    traj_y_concat = ca.vertcat(*traj_y)
    g = ca.vertcat(u)
    lb = ca.vertcat([-u_max] * T)
    ub = ca.vertcat([u_max] * T)

    prob = {'f': f_cost, 'x': u, 'g': g}
    solver_opts = {'ipopt': {'max_iter': 1000}}
    solver = ca.nlpsol('solver', 'ipopt', prob, solver_opts)

    # Get the optimal solution
    sol = solver(x0=0, lbg=lb, ubg=ub)
    u_sol = sol['x']

    px, py = start_pt[0], start_pt[1]
    px_sol = np.zeros((u_sol.shape[0]+1,1))
    px_sol[0] = px
    py_sol = np.zeros((u_sol.shape[0]+1,1))
    py_sol[0] = py

    for k in range(T):
        next = f(px, py, u_sol[k])
        px = next[0]
        py = next[1]
        px_sol[k+1] = px
        py_sol[k+1] = py

    return px_sol, py_sol, u_sol


## ---------------------------- FORWARD NONLINEAR OPTIMAL CONTROL ----------------------------

# Symbols/expressions
x1 = ca.MX.sym('x1')
x2 = ca.MX.sym('x2')
u = ca.MX.sym('u')

a = 0.9
b = 1
c = 0.9
d = 1

# State dynamics
f = ca.vertcat(a*x1+b*u, c*x2 + (c-a**2)*x1**2 + d*u)
f_x = ca.jacobian(f, ca.vertcat(x1,x2))
f_u = ca.jacobian(f, u)

# Cost
phi = ca.vertcat(x1**2, (x2+x1**2)**2, x1**4, x1**2*u**2, u**2, u**4)
phi_x = ca.jacobian(phi, ca.vertcat(x1, x2))
phi_u = ca.jacobian(phi, u)


# Weights
Wx1, Wx2, Wx3, Wux, Wu1, Wu2 = 1, 2, 3, 4, 5, 1

# Vector of weights to be learned
W = np.array([Wx1, Wx2, Wx3, Wux, Wu1, Wu2])

# Maximum discrete trajectory points
trajectory_length = 20

# Number of trajectories to generate (training set) + 1 (test set)
M = 1

# Optimal LQR state and input
x_star = np.zeros((M,trajectory_length,2))
u_star = np.zeros((M,trajectory_length,1))

start_pt = np.zeros((M,2))

for i in range(M):
    #start_pt[i,:] = np.matrix(10*np.random.rand(2)-5)
    start_pt[i,:] = np.array((5, 5))
    traj_x1,traj_x2,traj_u = generateTrajectory(start_pt[i,:], trajectory_length, W, 
        ca.Function('phi', [x1, x2, u], [phi]), ca.Function('f', [x1, x2, u], [f]))
    
    x_star[i,:,0] = traj_x1[:-1].reshape(trajectory_length)
    x_star[i,:,1] = traj_x2[:-1].reshape(trajectory_length)
    u_star[i,:,:] = traj_u

# Plot the actual trajectory for the test set
plt.plot(traj_x1, traj_x2, 'b-o', label='Nonlinear trajectory')


## ---------------------------- FORWARD LQR IN LIFTED STATE SPACE ----------------------------

def dlqr(A,B,Q,R,N,n):
    """
    Solve the discrete time lqr controller.
    x[k+1] = A x[k] + B u[k]
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k] + 2x[k].T*N*u[k]
    """
    P = [None]*n
    P[n-1] = np.zeros(A.shape)
    # For i = n, ..., 1
    for i in range(n-1, 0, -1):
        # Discrete-time Algebraic Riccati equation to calculate the optimal 
        # state cost matrix
        P[i-1] = Q + A.T @ P[i] @ A - (A.T @ P[i] @ B + N) @ np.linalg.pinv(
            R + B.T @ P[i] @ B) @ (B.T @ P[i] @ A + N.T)      
 
    # Create a list of N elements
    K = [None]*(n-1)
 
    # For i = 0, ..., n - 1
    for i in range(n-1):
        # Calculate the optimal feedback gain K
        K[i] = -np.linalg.pinv(R + B.T @ P[i+1] @ B) @ (B.T @ P[i+1] @ A + N.T)
    
    return K


A_hat = np.matrix([[a, 0, 0, 0],
                   [0, c, 0, 0],
                   [0, 0, a**2, 0],
                   [0, 0, 0, 0]])
B_hat = np.matrix([[0, b, 0, 0],
                   [2*a*b, d, b**2, 0],
                   [2*a*b, 0, b**2, 0],
                   [0, 0, 0, 1]])

ctrb = np.hstack(
        [B_hat] + [np.linalg.matrix_power(A_hat, i) @ B_hat
                  for i in range(1, A_hat.shape[0])])
print("Dims of controllability matrix: " + str(ctrb.shape))
print("Rank of controllability matrix: " + str(np.linalg.matrix_rank(ctrb)))

Q_hat = np.matrix([[Wx1, 0, 0, 0],
                   [0, Wx2, 0, 0],
                   [0, 0, Wx3, 0],
                   [0, 0, 0, 0]])
R_hat = np.matrix([[Wux, 0, 0, 0],
                   [0, Wu1, 0, 0],
                   [0, 0, Wu2, 0],
                   [0, 0, 0, 0]])

p = 1
R_tilde = np.matrix([[0, 0, 0, 0],
                     [0, -1, 0, 0],
                     [0, 0, 0, 0],
                     [0, 0, 1, 0]])
N_tilde = np.matrix([[0, -1/2, 0, 0],
                     [0, 0, 0, 0],
                     [0, 0, 0, 0],
                     [1/2, 0, 0, 0]])

nsteps = trajectory_length
K_hat = dlqr(A_hat,B_hat,Q_hat,R_hat+p*R_tilde,p*N_tilde,nsteps)

# Optimal LQR state and input in error state
Z_hat_star = []; V_hat_star = []
for m in range(M):
    z_hat_star = np.zeros((nsteps,4))
    v_hat_star = np.zeros((nsteps-1,4))

    # Transform initial states in (x,u) to (z_hat,v_hat)
    z_hat_k = np.array((start_pt[m,0], start_pt[m,1]+start_pt[m,0]**2, start_pt[m,0]**2, 1)).reshape(4,1)

    # Convert initial state to error state
    z_hat_ref = np.array((0, 0, 0, 1)).reshape(4,1)
    Dz_hat_k = z_hat_k - z_hat_ref

    # Reference controller to achieve z_hat_ref
    v_hat_ref = np.array((0, 0, 0, 1)).reshape(4,1)

    # Save initial optimal state
    z_hat_star[0,:] = z_hat_k.ravel()

    for i in range(len(K_hat)):
        v_hat_k = K_hat[i] @ Dz_hat_k + v_hat_ref
        Dv_hat_k = v_hat_k - v_hat_ref
        Dz_hat_k = A_hat @ Dz_hat_k + B_hat @ Dv_hat_k
        z_hat_k = Dz_hat_k + z_hat_ref
        z_hat_star[i+1,:] = z_hat_k.ravel()
        v_hat_star[i,:] = v_hat_k.ravel()
    
    Z_hat_star.append(z_hat_star); V_hat_star.append(v_hat_star)

Z_hat_star = np.array(Z_hat_star)
V_hat_star = np.array(V_hat_star)

# Convert trajectory in lifted states (z_hat, v_hat) to (x,u)
C = np.matrix([[1, 0, 0, 0],
               [0, 1, -1, 0]])
x_hat_star = C @ z_hat_star.T
x_hat_star = x_hat_star.T

print("z_hat:")
print(z_hat_star)
print("v_hat")
print(v_hat_star)

# Plot LQR trajectory in original states (x,u)
plt.plot(x_hat_star[:,0], x_hat_star[:,1], 'g-o', label='LQR trajectory')
plt.legend(loc='lower right')
plt.grid()
plt.show()