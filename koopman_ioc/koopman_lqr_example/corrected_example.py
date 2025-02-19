import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
from tqdm import tqdm

from nIOC import *
from iLQR import *

import sympy as sp
from sympy.polys.orderings import monomial_key
from sympy.polys.monomials import itermonomials


# Supress scientific notation when printing NumPy arrays
np.set_printoptions(precision=5,suppress=True)


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
phi = ca.vertcat(x1**2, (x2+x1**2)**2, x1**4, x1**2*u**2, u**4, u**2)
phi_x = ca.jacobian(phi, ca.vertcat(x1, x2))
phi_u = ca.jacobian(phi, u)


# Weights
Wx1, Wx2, Wx3, Wux, Wu1, Wu2 = 1, 2, 3, 1, 1, 1

# Vector of weights to be learned
W = np.array([Wx1, Wx2, Wx3, Wux, Wu1, Wu2])

# Maximum discrete trajectory points
trajectory_length = 30

# Number of trajectories to generate (training set) + 1 (test set)
M = 500

# Optimal LQR state and input
x_star = np.zeros((M,trajectory_length+1,2))
u_star = np.zeros((M,trajectory_length,1))

start_pt = np.zeros((M,2))

for i in tqdm(range(M)):
    start_pt[i,:] = np.matrix(10*np.random.rand(2)-5)
    #start_pt[i,:] = np.array((5, 5))
    traj_x1,traj_x2,traj_u = generateTrajectory(start_pt[i,:], trajectory_length, W, 
        ca.Function('phi', [x1, x2, u], [phi]), ca.Function('f', [x1, x2, u], [f]))
    
    x_star[i,:,0] = traj_x1.reshape(trajectory_length+1)
    x_star[i,:,1] = traj_x2.reshape(trajectory_length+1)
    u_star[i,:,:] = traj_u

s = x_star
c = u_star

# Plot the actual trajectory for the test set
plt.plot(traj_x1, traj_x2, 'b-o', label='Actual trajectory')


## ---------------------------- INVERSE NONLINEAR OPTIMAL CONTROL ----------------------------

traj_len = trajectory_length
states = s[-1,:traj_len,:]
controls = c[-1,:traj_len,:]

M_ = M_fun(states,controls, ca.Function('phi_x', [x1, x2, u], [phi_x]))
N_ = N_fun(states,controls, ca.Function('phi_u', [x1, x2, u], [phi_u]))
A_ = A_fun(states,controls, ca.Function('f_x', [x1, x2, u], [f_x]))
B_ = B_fun(states,controls, ca.Function('f_u', [x1, x2, u], [f_u]))
V_ = V_fun(states,controls, ca.Function('f_x', [x1, x2, u], [f_x]))

# Calculate F and E matrices
F = ca.mtimes(ca.mtimes(B_, ca.inv(A_)), M_) + N_
E = ca.mtimes(ca.mtimes(B_, ca.inv(A_)), V_)

# Check for IOC effectiveness
if np.linalg.matrix_rank(E) == states.shape[1]:
    R = F - ca.mtimes(ca.mtimes(E, ca.inv(ca.mtimes(E.T, E))), ca.mtimes(E.T, F))
else:
    print("Not IOC effective")

# Solve centralized, single-agent problem
z = ca.SX.sym('z', phi.shape[0])
z_sum = ca.vertcat(z[0])
lb = ca.vertcat(1)
ub = ca.vertcat(1)

# Define the optimization problem
f_obj = 1/2 * ca.mtimes(z.T, ca.mtimes(ca.mtimes(R.T, R), z))**2
prob = {'f': f_obj, 'x': z, 'g': z_sum}
solver_opts = {'ipopt': {'max_iter': 1000}}
solver = ca.nlpsol('solver', 'ipopt', prob, solver_opts)

# Solve the optimization problem
sol = solver(x0=0, lbg=lb, ubg=ub)

# Generate trajectory using the learned weights
weights_learned = sol['x']
weights_normalized = W / np.linalg.norm(W)
weights_normalized_bias = weights_normalized * sol['x'][0] / weights_normalized[0]

traj_x, traj_y, traj_u = generateTrajectory(start_pt[M-1,:], trajectory_length, weights_learned, 
        ca.Function('phi', [x1, x2, u], [phi]), ca.Function('f', [x1, x2, u], [f]))

# Plot the estimated trajectory for the test set
plt.plot(traj_x, traj_y, 'k-o', label='nIOC Estimated trajectory')


## Extra data to assess solution
# Add a trajectory from a random Q to the plot of the test set
weights_random = np.random.rand(*W.shape)
traj_x_random, traj_y_random, _ = generateTrajectory(start_pt[M-1,:], 
    trajectory_length, weights_random, ca.Function('phi', [x1, x2, u], [phi]), 
    ca.Function('f', [x1, x2, u], [f]))

plt.plot(traj_x_random, traj_y_random, 'g-o', label='Random trajectory')

# Show original normalized weights and the learned weights
print("Learned weights | Original normalized weights")
print(np.hstack([weights_learned, weights_normalized.reshape(1,-1).T]))

# Show original normalized weights (without bias) and the learned weights
print("Learned weights | Original normalized weights (without bias)")
print(np.hstack([weights_learned, weights_normalized_bias]))

plt.legend(loc='lower right')
plt.grid()
plt.show()


## ---------------------------- KOOPMAN LINEAR RE-REFORMULATION ----------------------------


x1, x2, u = sp.symbols('x1 x2 u', real=True)

deg_theta = 2
monomials_theta = sorted(itermonomials([x1, x2], deg_theta), key=monomial_key('grlex', [x2, x1]))
monomials_theta = monomials_theta[1:]
monomials_theta = monomials_theta[0:3] # To fit example dimensions perfectly
basis_theta = sp.lambdify((x1,x2),monomials_theta,"numpy")

deg_psi = 2
monomials_psi = sorted(itermonomials([x1, x2, u], deg_psi), key=monomial_key('grlex', [x2, x1, u]))
monomials_psi = monomials_psi[1:]
monomials_psi = [monomials_psi[4], monomials_psi[0], monomials_psi[3]] # To fit example dimensions perfectly
basis_psi = sp.lambdify((x1,x2,u),monomials_psi,"numpy")

print(monomials_theta)
print(monomials_psi)

# Dictionary of monomials for EDMD, deg = 3 i.e. 20 total dictionary functions
def basis_theta_fun(X):
    ret = np.array(basis_theta(X[:,0],X[:,1]))
    return ret

# Dictionary of monomials for EDMD, deg = 3 i.e. 20 total dictionary functions
def basis_psi_fun(X,U):
    ret = np.array(basis_psi(X[:,0],X[:,1],U[:,0]))
    return ret

# EDMD
X = x_star[:,:-1,:].reshape(M*trajectory_length,-1)
Z = basis_theta_fun(X)

U = u_star.reshape(M*trajectory_length,-1)
V = basis_psi_fun(X,U)

Y = x_star[:,1:,:].reshape(M*trajectory_length,-1)
Z_prime = basis_theta_fun(Y)

AB = np.matmul(Z_prime,np.linalg.pinv(np.vstack((Z, V))))
A = AB[:,:len(monomials_theta)]
B = AB[:,len(monomials_theta):len(monomials_theta)+len(monomials_psi)]
print("Dims of original AB : " + str(AB.shape))
print("Rank of AB : " + str(np.linalg.matrix_rank(AB)))
print("Dims of original A : " + str(A.shape))
print("Rank of A : " + str(np.linalg.matrix_rank(A)))
print("Dims of original B : " + str(B.shape))
print("Rank of B : " + str(np.linalg.matrix_rank(B)))

"""
## Sorting process

Z_ext = np.vstack((Z, V))
Fphi = np.eye(Z_ext.shape[0])
Ztr = Z_ext.copy()
Ztr_prime = Z_prime.copy()

while(Ztr.shape[0]>6):
    AB = np.matmul(Ztr_prime,np.linalg.pinv(Ztr))
    mu,phi = np.linalg.eig(AB.T)
    res_eig = np.abs(np.matmul(phi.T,Ztr_prime - np.matmul(AB,Ztr)))
    res_eig = np.max(res_eig,axis=1)
    #print(np.sort(res_eig))
    s = np.argsort(res_eig)
    phi = phi[:,s]
    m = Ztr.shape[0]
    m = int(m/2)
    phi = phi[:,0:m]
    Fphi = np.matmul(Fphi,phi) 
    Ztr = np.matmul(Fphi.T,Z_ext)
    Ztr_prime = np.matmul(Fphi.T,Z_prime)

AB = np.matmul(Ztr_prime,np.linalg.pinv(Ztr))

mu,phi = np.linalg.eig(AB.T)
lam = mu
K = np.diag(mu)
phi = np.matmul(Fphi,phi)
res_eig = np.abs(np.matmul(phi.T,Z_prime) - np.matmul(K.T,np.matmul(phi.T,Z_ext)))
res_eig = np.max(res_eig,axis=1)
#print(np.sort(res_eig))

# Get the stable eigenmodes
stable = np.argwhere(np.real(lam)<0).T[0]
phi_stable = phi[:,stable]
lam_stable = lam[stable]
sort_idx = np.argsort(np.real(lam_stable))
phi_stable = phi_stable[:,sort_idx]
lam_stable = lam_stable[sort_idx]

g = sp.Matrix(np.real(phi_stable.T)*(abs(np.real(phi_stable.T))>1e-3))*sp.Matrix(monomials[1:])
print(g)

A = AB[:,:len(monomials_theta)]
B = AB[:,len(monomials_theta):len(monomials_theta)+len(monomials_psi)]
print("Dims of sorted AB : " + str(AB.shape))
print("Dims of sorted A : " + str(A.shape))
print("Dims of sorted B : " + str(B.shape))
"""

res = np.linalg.norm(Z_prime - np.matmul(AB,np.vstack((Z, V))))**2
print("Residual of EDMDc : " + str(res))

with np.printoptions(precision=4, suppress=True, formatter={'float': '{:0.4f}'.format}, linewidth=200):
    print("Computed A: ")
    print(A)
    print("Computed B: ")
    print(B)


## ---------------------------- INVERSE LQR ----------------------------

# Inverse LQR
Z = Z.T.reshape(M,trajectory_length,-1)
V = V.T.reshape(M,trajectory_length,-1)

print("Dims of Z : " + str(Z.shape))
print("Dims of V : " + str(V.shape))

ctrb = np.hstack(
        [B] + [np.linalg.matrix_power(A, i) @ B
                  for i in range(1, A.shape[0])])
print("Dims of controllability matrix: " + str(ctrb.shape))
print("Rank of controllability matrix: " + str(np.linalg.matrix_rank(ctrb)))

Q_re = iLQR(A, B, Z, V)

print("Computed Q: ")
with np.printoptions(precision=4, suppress=True, formatter={'float': '{:0.4f}'.format}, linewidth=200):
    print(Q_re)


## ---------------------------- Forward LQR to ORIGINAL STATES ----------------------------

def dlqr(A,B,Q,R,n):
    """
    Solve the discrete time lqr controller.
    x[k+1] = A x[k] + B u[k]
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    """
    P = [None]*n
    P[n-1] = np.zeros(A.shape)
    # For i = n, ..., 1
    for i in range(n-1, 0, -1):
        # Discrete-time Algebraic Riccati equation to calculate the optimal 
        # state cost matrix
        P[i-1] = Q + A.T @ P[i] @ A - (A.T @ P[i] @ B) @ np.linalg.pinv(
            R + B.T @ P[i] @ B) @ (B.T @ P[i] @ A)      
 
    # Create a list of N elements
    K = [None]*(n-1)
 
    # For i = 0, ..., n - 1
    for i in range(n-1):
        # Calculate the optimal feedback gain K
        K[i] = -np.linalg.pinv(R + B.T @ P[i+1] @ B) @ (B.T @ P[i+1] @ A)
    
    return K

R = np.identity(B.shape[1])

nsteps = trajectory_length
K = dlqr(A,B,Q_re,R,nsteps)

# Optimal LQR state and input for final trajectory
j = M-1
z_lqr_star = np.zeros((nsteps,Z.shape[2]))
v_lqr_star = np.zeros((nsteps-1,V.shape[2]))

# Transform initial states in (x,u) to (z_lqr,v_lqr)
z_lqr_k = Z[j,0,:]

# Save initial optimal state
z_lqr_star[0,:] = z_lqr_k.ravel()

for i in range(len(K)):
    v_lqr_k = K[i] @ z_lqr_k
    z_lqr_k = A @ z_lqr_k + B @ v_lqr_k
    z_lqr_star[i+1,:] = z_lqr_k.ravel()
    v_lqr_star[i,:] = v_lqr_k.ravel()

# Convert trajectory in lifted states (z_lqr, v_lqr) to (x,u)
C = np.zeros((2, Z.shape[2]))
C[0,0] = 1 
C[1,1] = 1
x_lqr_star = C @ z_lqr_star.T
x_lqr_star = x_lqr_star.T

# Plot LQR trajectory in original states (x,u)
plt.plot(x_lqr_star[:,0], x_lqr_star[:,1], 'g-o', label='LQR trajectory')
# Plot the nonlinear trajectory
plt.plot(traj_x1, traj_x2, 'b-o', label='Nonlinear trajectory')
plt.legend(loc='lower right')
plt.grid()
plt.show()