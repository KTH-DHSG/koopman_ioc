import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import casadi as ca
from tqdm import tqdm
import os

from nIOC import *
from iBiPMP_withR import *

import sympy as sp
from sympy.polys.orderings import monomial_key
from sympy.polys.monomials import itermonomials

from sklearn.linear_model import MultiTaskLasso


# Supress scientific notation when printing NumPy arrays
np.set_printoptions(precision=5,suppress=True)


## ------------------------------ FORWARD NONLINEAR OPTIMAL CONTROL ------------------------------

# Symbols/expressions
x1 = ca.MX.sym('x1')
x2 = ca.MX.sym('x2')
x3 = ca.MX.sym('x3')
u1 = ca.MX.sym('u1')
u2 = ca.MX.sym('u2')

# Sampling time dt < 1
dt = 1/50 # 100 Hz

# State dynamics
f = ca.vertcat(x1 + dt*u1*ca.cos(x3), x2 + dt*u1*ca.sin(x3), x3 + dt*u2)
f_x = ca.jacobian(f, ca.vertcat(x1,x2,x3))
f_u = ca.jacobian(f, ca.vertcat(u1,u2))

# Obstacle selection
obstacles = None
# obstacles = np.array([[-1, -1]])
# obstacles = np.array([[-1, -1], [-1, -2], [-2, -2]])

# Cost
phi = ca.vertcat(x1**2, x2**2,  x3**2, ca.cos(x3)**2, ca.sin(x3)**2, 1, u1**2, u2**2)
if obstacles is not None:
    for i in range(obstacles.shape[0]):
        phi = ca.vertcat(phi, ca.exp(-10 * ((x1 - obstacles[i, 0])**2 + (x2 - obstacles[i, 1])**2)))
phi_x = ca.jacobian(phi, ca.vertcat(x1, x2, x3))
phi_u = ca.jacobian(phi, ca.vertcat(u1, u2))

# Weights
Wx1, Wx2, Wx3, Wcos, Wsin, W1, Wu1, Wu2, Wobs = 1, 1, 1, 0, 0, 0, 1, 1, 2

# Vector of weights to be learned
if obstacles is not None:
    W = np.array([Wx1, Wx2, Wx3, Wcos, Wsin, W1, Wu1, Wu2] + [Wobs] * obstacles.shape[0])
else:
    W = np.array([Wx1, Wx2, Wx3, Wcos, Wsin, W1, Wu1, Wu2])

# Maximum discrete trajectory points
trajectory_length = 200

# Number of trajectories to generate (training set) + 1 (test set)
M = 30

# Generate filename
obs_shape = obstacles.shape[0] if obstacles is not None else 0
filename = f'koopman_ioc/bilinear_pmp_example/hmp_example/traj_data_{dt*1000}_{M}_{trajectory_length}_{obs_shape}.csv'

# Check if file exists
if os.path.exists(filename):
    # Import data from the CSV file
    data = pd.read_csv(filename)
    start_pt = data.filter(regex='^start').values.reshape((M, 3))
    x_star = data.filter(regex='^x').values.reshape((M, trajectory_length + 1, 3))
    u_star = data.filter(regex='^u').values.reshape((M, trajectory_length, 2))
else:
    # Optimal LQR state and input
    x_star = np.zeros((M,trajectory_length+1,3))
    u_star = np.zeros((M,trajectory_length,2))

    start_pt = np.zeros((M,3))

    for i in tqdm(range(M)):
        start_pt[i,:] = np.array([[2.5*np.random.rand(1)-5, 2.5*np.random.rand(1)-5, np.pi*np.random.rand(1)-np.pi]]).flatten()
        traj_x1,traj_x2,traj_th,traj_u1,traj_u2 = generateTrajectory(start_pt[i,:], trajectory_length, W, 
            ca.Function('phi', [x1, x2, x3, u1, u2], [phi]), ca.Function('f', [x1, x2, x3, u1, u2], [f]))
        
        x_star[i,:,0] = traj_x1.reshape(trajectory_length+1)
        x_star[i,:,1] = traj_x2.reshape(trajectory_length+1)
        x_star[i,:,2] = traj_th.reshape(trajectory_length+1)
        u_star[i,:,0] = traj_u1.reshape(trajectory_length)
        u_star[i,:,1] = traj_u2.reshape(trajectory_length)

    # Prepare data for DataFrame
    data_dict = {}
    
    for i in range(M):
        data_dict[f'start_{i}_0'] = start_pt[i, 0]
        data_dict[f'start_{i}_1'] = start_pt[i, 1]
        data_dict[f'start_{i}_2'] = start_pt[i, 2]
        for j in range(trajectory_length + 1):
            data_dict[f'x_{i}_{j}_0'] = x_star[i, j, 0]
            data_dict[f'x_{i}_{j}_1'] = x_star[i, j, 1]
            data_dict[f'x_{i}_{j}_2'] = x_star[i, j, 2]
        for j in range(trajectory_length):
            data_dict[f'u_{i}_{j}_0'] = u_star[i, j, 0]
            data_dict[f'u_{i}_{j}_1'] = u_star[i, j, 1]

    # Convert dictionary values to lists
    data_dict = {key: [value] for key, value in data_dict.items()}
    
    # Create DataFrame from dictionary
    data = pd.DataFrame(data_dict)

    # Save DataFrame to CSV
    data.to_csv(filename, index=False)

s = x_star
cont = u_star

# Plot the actual trajectory for the test set
plt.plot(x_star[-1, :, 0], x_star[-1, :, 1], 'b-o', label='Actual trajectory')

# Plot obstacles if they exist
if obstacles is not None and not np.all(obstacles == 0):
    plt.plot(obstacles[:, 0], obstacles[:, 1], '*r', label='Obstacles')


## ------------------------------ INVERSE NONLINEAR OPTIMAL CONTROL ------------------------------

traj_len = trajectory_length
states = s[-1,:traj_len,:]
controls = cont[-1,:traj_len,:]

M_ = M_fun(states,controls, ca.Function('phi_x', [x1, x2, x3, u1, u2], [phi_x]))
N_ = N_fun(states,controls, ca.Function('phi_u', [x1, x2, x3, u1, u2], [phi_u]))
A_ = A_fun(states,controls, ca.Function('f_x', [x1, x2, x3, u1, u2], [f_x]))
B_ = B_fun(states,controls, ca.Function('f_u', [x1, x2, x3, u1, u2], [f_u]))
V_ = V_fun(states,controls, ca.Function('f_x', [x1, x2, x3, u1, u2], [f_x]))

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

traj_x, traj_y, _, traj_u1, traj_u2 = generateTrajectory(start_pt[M-1,:], trajectory_length, weights_learned, 
        ca.Function('phi', [x1, x2, x3, u1, u2], [phi]), ca.Function('f', [x1, x2, x3, u1, u2], [f]))

# Plot the estimated trajectory for the test set
plt.plot(traj_x, traj_y, 'k-o', label='nIOC Estimated trajectory')


## Extra data to assess solution
# Add a trajectory from a random Q to the plot of the test set
weights_random = np.random.rand(*W.shape)
traj_x_random, traj_y_random, _, _, _ = generateTrajectory(start_pt[M-1,:], 
    trajectory_length, weights_random, ca.Function('phi', [x1, x2, x3, u1, u2], [phi]), 
    ca.Function('f', [x1, x2, x3, u1, u2], [f]))

plt.plot(traj_x_random, traj_y_random, 'm-o', label='Random trajectory')

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

x1, x2, x3 = sp.symbols('x1 x2 x3', real=True)

deg = 1
monomials = sorted(itermonomials([x1, x2, x3, sp.cos(x3), sp.sin(x3)], deg), key=monomial_key('grlex', [sp.sin(x3), sp.cos(x3), x3, x2, x1]))
basis = sp.lambdify((x1,x2,x3),monomials,"numpy")

print(monomials)

# Dictionary of monomials for EDMDc, deg = 3 i.e. 20 total dictionary functions
def basis_fun(X):
    ret = np.vstack((np.ones((1,X.shape[0])),np.array(basis(X[:,0],X[:,1],X[:,2])[1:]))) 
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

'''
## Sorting process
Fphi = np.eye(Z.shape[0])
Ztr = Z.copy()
bi_Utr = bi_U.copy()
Ztr_prime = Z_prime.copy()

while(Ztr.shape[0]>6):
    # Compute new A and B matrices
    AB = np.matmul(Ztr_prime,np.linalg.pinv(np.vstack((Ztr, bi_Utr))))
    A = AB[:,:Ztr.shape[0]]
    B = AB[:,Ztr.shape[0]:]
    # Compute the eigenvalues/vectors of A and sort Z with them
    mu,phi = np.linalg.eig(A.T)
    res_eig = np.abs(np.matmul(phi.T,Ztr_prime - np.matmul(A,Ztr) - np.matmul(np.matmul(B[:,:Ztr.shape[0]],np.linalg.pinv(phi.T)),bi_U1) - 
                               np.matmul(np.matmul(B[:,Ztr.shape[0]:],np.linalg.pinv(phi.T)),bi_U2)))
    res_eig = np.max(res_eig,axis=1)
    #print(np.sort(res_eig))
    s = np.argsort(res_eig)
    phi = phi[:,s]
    m = Ztr.shape[0]
    m -= 1
    phi = phi[:,0:m]
    Fphi = np.matmul(Fphi,phi)
    # Recompute data (Ztr_prime, Ztr, bi_Utr)
    Ztr = np.matmul(Fphi.T,Z)
    bi_U1 = Ztr * np.ones((Ztr.shape[0],1))*U1
    bi_U2 = Ztr * np.ones((Ztr.shape[0],1))*U2
    bi_Utr = np.vstack((bi_U1, bi_U2))
    Ztr_prime = np.matmul(Fphi.T,Z_prime)

bi_U1 = np.abs(Ztr)**2 * np.ones((Ztr.shape[0],1))*U1
bi_U2 = np.abs(Ztr)**2 * np.ones((Ztr.shape[0],1))*U2
bi_Utr = np.vstack((bi_U1, bi_U2))
AB = np.matmul(np.abs(Ztr_prime)**2,np.linalg.pinv(np.vstack((np.abs(Ztr)**2, bi_Utr))))
A = AB[:,:Ztr.shape[0]]
B = AB[:,Ztr.shape[0]:]
'''

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
A_hat = np.matrix([[1, 0, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0, 0],
                   [0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 0, 1]])
B_hat = np.matrix([[0, 0, 0, dt, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, dt, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, dt],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -dt, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, dt, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

ctrb = np.hstack(
        [B] + [np.linalg.matrix_power(A, i) @ B
                  for i in range(1, A.shape[0])])
print("Dims of controllability matrix: " + str(ctrb.shape))
print("Rank of controllability matrix: " + str(np.linalg.matrix_rank(ctrb)))

Q_re, R_re = iBiPMP_withR(A, B, Z, U)

with np.printoptions(precision=4, suppress=True, formatter={'float': '{:0.4f}'.format}, linewidth=200):
    print("Computed Q: ")
    print(Q_re)
    print("Computed R: ")
    print(R_re)


# ## ----------------------------- FORWARD BILINEAR OPT CONTROL PROBLEM -----------------------------

# # Symbols/expressions
# z = ca.MX.sym('x',len(monomials))
# u1 = ca.MX.sym('u1')
# u1 = ca.MX.sym('u2')

# # Correct/Expect Cost
# Q_hat = np.matrix([[Wx1, 0, 0, 0, 0, 0],
#                    [0, Wx2, 0, 0, 0, 0],
#                    [0, 0, Wx3, 0, 0, 0],
#                    [0, 0, 0, Wcos, 0, 0],
#                    [0, 0, 0, 0, Wsin, 0],
#                    [0, 0, 0, 0, 0, W1]])

# # Lifted State dynamics
# f = A @ z + B[:,:len(monomials)] @ z * u1 + B[:,len(monomials):] @ z * u2
# f_z = ca.jacobian(f, z)
# f_u = ca.jacobian(f, ca.vertcat(u1,u2))

# # Cost
# R = np.identity(2)
# phi = z.T @ Q_re @ z + ca.vertcat(u1,u2).T @ R @ ca.vertcat(u1,u2)
# phi_z = ca.jacobian(phi, z)
# phi_u = ca.jacobian(phi, ca.vertcat(u1,u2))

# lifted_start_point = basis(start_pt[-1,0], start_pt[-1,1], start_pt[-1,2])

# # Compute optimal trajectory
# traj_z,_,_ = generateLiftedTrajectory(lifted_start_point, trajectory_length, 
#     ca.Function('phi', [z, u1, u2], [phi]), ca.Function('f', [z, u1, u2], [f]), len(monomials))

# #print("Computed Z: ")
# #print(traj_z)

# # Convert trajectory in lifted states (z,u) to (x,u)
# x_bi_star = C @ traj_z.T
# x_bi_star = x_bi_star.T

# # Plot LQR trajectory in original states (x,u)
# plt.plot(x_bi_star[:,0], x_bi_star[:,1], 'g-o', label='Bilinear trajectory')
# # Plot obstacles if they exist
# if obstacles is not None and not np.all(obstacles == 0):
#     plt.plot(obstacles[:, 0], obstacles[:, 1], '*r', label='Obstacles')
# # Plot the nonlinear trajectory
# plt.plot(x_star[-1, :, 0], x_star[-1, :, 1], 'b-+', label='Nonlinear trajectory')
# plt.plot(traj_x, traj_y, 'k-x', label='nIOC Estimated trajectory')
# plt.legend(loc='lower right')
# plt.grid()
# plt.show()


## ----------------------------- FORWARD NONLINEAR OPT CONTROL PROBLEM -----------------------------

# Symbols/expressions
x1 = ca.MX.sym('x1')
x2 = ca.MX.sym('x2')
x3 = ca.MX.sym('x3')
u1 = ca.MX.sym('u1')
u2 = ca.MX.sym('u2')

# Convert the SymPy expression to a string
monomial_strings = [str(m) for m in monomials]

# Create a dictionary for CasADi variables
casadi_vars = {'x1': x1, 'x2': x2, 'x3': x3}

# Parse the string with CasADi
ca_basis = []
for m_str in monomial_strings:
    # Replace sympy functions with casadi functions
    m_str = m_str.replace('sin', 'ca.sin').replace('cos', 'ca.cos')
    # Use eval to convert the string to a casadi expression
    casadi_expr = eval(m_str, {"ca": ca}, casadi_vars)
    ca_basis.append(casadi_expr)

# Convert list of expressions to a casadi vertical concatenated vector
ca_basis = ca.vertcat(*ca_basis)

# Correct/Expect Cost
Q_hat = np.matrix([[Wx1, 0, 0, 0, 0, 0],
                   [0, Wx2, 0, 0, 0, 0],
                   [0, 0, Wx3, 0, 0, 0],
                   [0, 0, 0, Wcos, 0, 0],
                   [0, 0, 0, 0, Wsin, 0],
                   [0, 0, 0, 0, 0, W1]])

# Lifted State dynamics
f = C @ A @ ca_basis + C @ B[:,:len(monomials)] @ ca_basis * u1 + C @ B[:,len(monomials):] @ ca_basis * u2
f_x = ca.jacobian(f, ca.vertcat(x1,x2,x3))
f_u = ca.jacobian(f, ca.vertcat(u1,u2))

# Cost
# R = np.identity(2)
phi = ca_basis.T @ Q_re @ ca_basis + ca.vertcat(u1,u2).T @ R_re @ ca.vertcat(u1,u2)
phi_x = ca.jacobian(phi, ca.vertcat(x1, x2, x3))
phi_u = ca.jacobian(phi, ca.vertcat(u1, u2))

# Compute optimal trajectory
traj_x1,traj_x2,_,_,_ = generateTrajectory(start_pt[-1,:], trajectory_length, 1, 
    ca.Function('phi', [x1, x2, x3, u1, u2], [phi]), ca.Function('f', [x1, x2, x3, u1, u2], [f]))

# Plot LQR trajectory in original states (x,u)
plt.plot(traj_x1, traj_x2, 'g-o', label='Bilinear trajectory')
# Plot obstacles if they exist
if obstacles is not None and not np.all(obstacles == 0):
    plt.plot(obstacles[:, 0], obstacles[:, 1], '*r', label='Obstacles')
# Plot the nonlinear trajectory
plt.plot(x_star[-1, :, 0], x_star[-1, :, 1], 'k-x', label='Nonlinear trajectory')
# plt.plot(traj_x, traj_y, 'k-x', label='nIOC Estimated trajectory')
plt.plot(traj_x_random, traj_y_random, 'y-o', label='Random trajectory')
plt.legend(loc='lower right')
plt.grid()
plt.show()