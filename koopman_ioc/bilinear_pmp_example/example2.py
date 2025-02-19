import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
from tqdm import tqdm

from nIOC import *
from iBiPMP import *

import sympy as sp
from sympy.polys.orderings import monomial_key
from sympy.polys.monomials import itermonomials


# Supress scientific notation when printing NumPy arrays
np.set_printoptions(precision=5,suppress=True)


## ------------------------------ FORWARD NONLINEAR OPTIMAL CONTROL ------------------------------

# Symbols/expressions
x1 = ca.MX.sym('x1')
x2 = ca.MX.sym('x2')
u1 = ca.MX.sym('u1')
u2 = ca.MX.sym('u2')

# Sampling time dt < 1
dt = 0.01 # 100 Hz

# Continuous-time dynamics weights [0, 1]
c = 0.3
d = 0.2

# State dynamics
f = ca.vertcat((1+c*dt)*x1+dt*u1, (1+d*dt)*x2 + (d-2*c)*dt*x1**2 + dt*x1**2*u1 + dt*u2)
f_x = ca.jacobian(f, ca.vertcat(x1,x2))
f_u = ca.jacobian(f, ca.vertcat(u1,u2))

# Cost
phi = ca.vertcat(x1**2, x2**2, x1**4, 1, u1**2, u2**2)
phi_x = ca.jacobian(phi, ca.vertcat(x1,x2))
phi_u = ca.jacobian(phi, ca.vertcat(u1,u2))


# Weights
Wx1, Wx2, Wx1_2, W1, Wu1, Wu2 = 1, 2, 3, 1, 1, 1

# Vector of weights to be learned
W = np.array([Wx1, Wx2, Wx1_2, W1, Wu1, Wu2])

# Maximum discrete trajectory points
trajectory_length = 60

# Number of trajectories to generate (training set) + 1 (test set)
M = 1000

# Optimal LQR state and input
x_star = np.zeros((M,trajectory_length+1,2))
u_star = np.zeros((M,trajectory_length,2))

start_pt = np.zeros((M,2))

for i in tqdm(range(M)):
    start_pt[i,:] = np.matrix(10*np.random.rand(2)-5)
    #start_pt[i,:] = np.array((5, 5))
    traj_x1,traj_x2,traj_u1,traj_u2 = generateTrajectory(start_pt[i,:], trajectory_length, W, 
        ca.Function('phi', [x1, x2, u1, u2], [phi]), ca.Function('f', [x1, x2, u1, u2], [f]))
    
    x_star[i,:,0] = traj_x1.reshape(trajectory_length+1)
    x_star[i,:,1] = traj_x2.reshape(trajectory_length+1)
    u_star[i,:,0] = traj_u1.reshape(trajectory_length)
    u_star[i,:,1] = traj_u2.reshape(trajectory_length)

s = x_star
cont = u_star

# Plot the actual trajectory for the test set
plt.plot(traj_x1, traj_x2, 'b-o', label='Actual trajectory')


## ------------------------------ INVERSE NONLINEAR OPTIMAL CONTROL ------------------------------

traj_len = trajectory_length
states = s[-1,:traj_len,:]
controls = cont[-1,:traj_len,:]

M_ = M_fun(states,controls, ca.Function('phi_x', [x1, x2, u1, u2], [phi_x]))
N_ = N_fun(states,controls, ca.Function('phi_u', [x1, x2, u1, u2], [phi_u]))
A_ = A_fun(states,controls, ca.Function('f_x', [x1, x2, u1, u2], [f_x]))
B_ = B_fun(states,controls, ca.Function('f_u', [x1, x2, u1, u2], [f_u]))
V_ = V_fun(states,controls, ca.Function('f_x', [x1, x2, u1, u2], [f_x]))

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
solver_opts = {'ipopt': {'max_iter': 1000}, 'ipopt.print_level':0, 'print_time':0}
solver = ca.nlpsol('solver', 'ipopt', prob, solver_opts)

# Solve the optimization problem
sol = solver(x0=0, lbg=lb, ubg=ub)

# Generate trajectory using the learned weights
weights_learned = sol['x']
weights_normalized = W / np.linalg.norm(W)
weights_normalized_bias = weights_normalized * sol['x'][0] / weights_normalized[0]

traj_x, traj_y, traj_u1, traj_u2 = generateTrajectory(start_pt[M-1,:], trajectory_length, weights_learned, 
        ca.Function('phi', [x1, x2, u1, u2], [phi]), ca.Function('f', [x1, x2, u1, u2], [f]))

# Plot the estimated trajectory for the test set
plt.plot(traj_x, traj_y, 'k-o', label='nIOC Estimated trajectory')


## Extra data to assess solution
# Add a trajectory from a random Q to the plot of the test set
weights_random = np.random.rand(*W.shape)
traj_x_random, traj_y_random, _, _ = generateTrajectory(start_pt[M-1,:], 
    trajectory_length, weights_random, ca.Function('phi', [x1, x2, u1, u2], [phi]), 
    ca.Function('f', [x1, x2, u1, u2], [f]))

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


## ------------------------------- KOOPMAN LINEAR RE-REFORMULATION -------------------------------

x1, x2 = sp.symbols('x1 x2', real=True)

deg = 2
monomials = sorted(itermonomials([x1, x2], deg), key=monomial_key('grlex', [x2, x1]))
monomials = [monomials[1], monomials[2], monomials[3], monomials[0]] # To fit example dimensions perfectly
basis = sp.lambdify((x1,x2),monomials,"numpy")

print(monomials)

# Dictionary of monomials for EDMDc, deg = 3 i.e. 20 total dictionary functions
def basis_fun(X):
    ret = np.vstack((np.array(basis(X[:,0],X[:,1])[:-1]),np.ones((1,X.shape[0])))) 
    return ret

# EDMDc
X = x_star[:,:-1,:].reshape(M*trajectory_length,-1)
Z = basis_fun(X)

C = np.matmul(X.T,np.linalg.pinv(Z))
print("Dims of C : " + str(C.shape))
with np.printoptions(precision=4, suppress=True, formatter={'float': '{:0.4f}'.format}, linewidth=200):
    print("Computed C: ")
    print(C)

#U = u_star[:,:,:].reshape(M*trajectory_length,-1)
U1 = u_star[:,:,0].reshape(1,-1)
U2 = u_star[:,:,1].reshape(1,-1)

bi_U1 = Z * np.ones((len(monomials),1))*U1
bi_U2 = Z * np.ones((len(monomials),1))*U2
bi_U = np.vstack((bi_U1, bi_U2))

Y = x_star[:,1:,:].reshape(M*trajectory_length,-1)
Z_prime = basis_fun(Y)

AB = np.matmul(Z_prime,np.linalg.pinv(np.vstack((Z, bi_U))))
A = AB[:,:len(monomials)]
B = AB[:,len(monomials):]

"""
## Sorting process

Z_ext = np.vstack((Z, bi_U))
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
"""

print("Dims of original AB : " + str(AB.shape))
print("Rank of AB : " + str(np.linalg.matrix_rank(AB)))
print("Dims of original A : " + str(A.shape))
print("Rank of A : " + str(np.linalg.matrix_rank(A)))
print("Dims of original B : " + str(B.shape))
print("Rank of B : " + str(np.linalg.matrix_rank(B)))

res = np.linalg.norm(Z_prime - np.matmul(AB,np.vstack((Z, bi_U))))**2
print("Residual of EDMDc : " + str(res))

with np.printoptions(precision=4, suppress=True, formatter={'float': '{:0.4f}'.format}, linewidth=200):
    print("Computed A: ")
    print(A)
    print("Computed B: ")
    print(B)


## ---------------------------------------- INVERSE BiPMP ----------------------------------------

# Inverse LQR
X_full = x_star.reshape(M*(trajectory_length+1),-1)
Z_full = basis_fun(X_full).T.reshape(M,trajectory_length+1,-1)
Z = Z_full
U = u_star

print("Dims of Z : " + str(Z.shape))
print("Dims of U : " + str(U.shape))

# Correct/Expect Dynamics
A_hat = np.matrix([[1+c*dt, 0, 0, 0],
                   [0, 1+d*dt, (d-2*c)*dt, 0],
                   [0, 0, 1+2*c*dt, 0],
                   [0, 0, 0, 1]])
B_hat = np.matrix([[0, 0, 0, dt, 0, 0, 0, 0],
                   [0, 0, dt, 0, 0, 0, 0, dt],
                   [2*dt, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0]])

ctrb = np.hstack(
        [B] + [np.linalg.matrix_power(A, i) @ B
                  for i in range(1, A.shape[0])])
print("Dims of controllability matrix: " + str(ctrb.shape))
print("Rank of controllability matrix: " + str(np.linalg.matrix_rank(ctrb)))

Q_re = iBiPMP(A, B, Z, U)

print("Computed Q: ")
with np.printoptions(precision=4, suppress=True, formatter={'float': '{:0.4f}'.format}, linewidth=200):
    print(Q_re)


## ----------------------------- FORWARD BILINEAR OPT CONTROL PROBLEM -----------------------------

# Symbols/expressions
z = ca.MX.sym('x',len(monomials))
u1 = ca.MX.sym('u1')
u1 = ca.MX.sym('u2')

# Correct/Expect Cost
Q_hat = np.matrix([[Wx1, 0, 0, 0],
                   [0, Wx2, 0, 0],
                   [0, 0, Wx1_2, 0],
                   [0, 0, 0, W1]])

# Lifted State dynamics
f = A @ z + B[:,:len(monomials)] @ z * u1 + B[:,len(monomials):] @ z * u2
f_z = ca.jacobian(f, z)
f_u = ca.jacobian(f, ca.vertcat(u1,u2))

# Cost
R = np.identity(2)
phi = z.T @ Q_re @ z + ca.vertcat(u1,u2).T @ R @ ca.vertcat(u1,u2)
phi_z = ca.jacobian(phi, z)
phi_u = ca.jacobian(phi, ca.vertcat(u1,u2))

lifted_start_point = basis(start_pt[-1,0], start_pt[-1,1])

# Compute optimal trajectory
traj_z,_,_ = generateLiftedTrajectory(lifted_start_point, trajectory_length, 
    ca.Function('phi', [z, u1, u2], [phi]), ca.Function('f', [z, u1, u2], [f]))

#print("Computed Z: ")
#print(traj_z)

# Convert trajectory in lifted states (z,u) to (x,u)
x_bi_star = C @ traj_z.T
x_bi_star = x_bi_star.T

# Plot LQR trajectory in original states (x,u)
plt.plot(x_bi_star[:,0], x_bi_star[:,1], 'g-o', label='Bilinear trajectory')
# Plot the nonlinear trajectory
plt.plot(traj_x1, traj_x2, 'k-+', label='Nonlinear trajectory')
# plt.plot(traj_x, traj_y, 'k-x', label='nIOC Estimated trajectory')
plt.plot(traj_x_random, traj_y_random, 'y-o', label='Random trajectory')
plt.legend(loc='lower right')
plt.grid()
plt.show()