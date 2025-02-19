import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import casadi as ca
from tqdm import tqdm
import os

from nIOC import *
from iBiPMP import *

import sympy as sp
from sympy.polys.orderings import monomial_key
from sympy.polys.monomials import itermonomials

from sklearn.linear_model import MultiTaskLasso

# Suppress scientific notation when printing NumPy arrays
np.set_printoptions(precision=5, suppress=True)


## ------------------------------ IMPORT REAL-WORLD DATA ------------------------------

filepath = 'koopman_ioc/bilinear_pmp_example/hmp_example/SML_experiment/data50hz_trimmed'

# Sampling time dt
dt = 1/50  # Adjust based on the data sampling rate

# Number of trajectories to use (training set)
M = 46

# Fixed trajectory length (adjust as per the data)
trajectory_length = 500  # Adjust according to the minimum, maximum or average length of the trajectories

# Initialize lists to hold data
x_star = []
u_star = []
start_pt = []

# Read the training data from test0.csv to test29.csv
for i in range(M):
    filename = f'{filepath}/test{i}.csv'
    data = pd.read_csv(filename)
    x_pos = data['x_pos'].values
    y_pos = data['y_pos'].values
    orientation = data['orientation'].values
    u_linear = data['u_linear'].values
    u_angular = data['u_angular'].values

    # Ensure all arrays are the same length
    min_len = min(len(x_pos), len(u_linear), len(u_angular))
    x_pos = x_pos[:min_len]
    y_pos = y_pos[:min_len]
    orientation = orientation[:min_len]
    u_linear = u_linear[:min_len]
    u_angular = u_angular[:min_len]

    # Truncate or pad to trajectory_length
    if len(x_pos) >= trajectory_length:
        x_pos = x_pos[:trajectory_length]
        y_pos = y_pos[:trajectory_length]
        orientation = orientation[:trajectory_length]
        u_linear = u_linear[:trajectory_length]
        u_angular = u_angular[:trajectory_length]
    else:
        # Pad shorter trajectories with the last value
        pad_length = trajectory_length - len(x_pos)
        x_pos = np.pad(x_pos, (0, pad_length), 'edge')
        y_pos = np.pad(y_pos, (0, pad_length), 'edge')
        orientation = np.pad(orientation, (0, pad_length), 'edge')
        u_linear = np.pad(u_linear, (0, pad_length), 'edge')
        u_angular = np.pad(u_angular, (0, pad_length), 'edge')

    states = np.vstack([x_pos, y_pos, orientation]).T  # shape (trajectory_length, 3)
    controls = np.vstack([u_linear, u_angular]).T      # shape (trajectory_length, 2)

    start_pt.append(states[0, :])  # initial state

    x_star.append(states)
    u_star.append(controls)

x_star = np.array(x_star)  # shape (M, trajectory_length, 3)
u_star = np.array(u_star)  # shape (M, trajectory_length, 2)
start_pt = np.array(start_pt)  # shape (M, 3)

# Read the test data from the last trajectory
filename = f'{filepath}/test26.csv'
data = pd.read_csv(filename)
x_pos = data['x_pos'].values
y_pos = data['y_pos'].values
orientation = data['orientation'].values
u_linear = data['u_linear'].values
u_angular = data['u_angular'].values

# Ensure all arrays are the same length
min_len = min(len(x_pos), len(u_linear), len(u_angular))
x_pos = x_pos[:min_len]
y_pos = y_pos[:min_len]
orientation = orientation[:min_len]
u_linear = u_linear[:min_len]
u_angular = u_angular[:min_len]

# Truncate or pad to trajectory_length
if len(x_pos) >= trajectory_length:
    x_pos = x_pos[:trajectory_length]
    y_pos = y_pos[:trajectory_length]
    orientation = orientation[:trajectory_length]
    u_linear = u_linear[:trajectory_length]
    u_angular = u_angular[:trajectory_length]
else:
    pad_length = trajectory_length - len(x_pos)
    x_pos = np.pad(x_pos, (0, pad_length), 'edge')
    y_pos = np.pad(y_pos, (0, pad_length), 'edge')
    orientation = np.pad(orientation, (0, pad_length), 'edge')
    u_linear = np.pad(u_linear, (0, pad_length), 'edge')
    u_angular = np.pad(u_angular, (0, pad_length), 'edge')

test_states = np.vstack([x_pos, y_pos, orientation]).T  # shape (trajectory_length, 3)
test_controls = np.vstack([u_linear, u_angular]).T      # shape (trajectory_length, 2)
test_start_pt = test_states[0, :]


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
X = x_star[:,:-1,:].reshape(M*(trajectory_length-1),-1)
Z = basis_fun(X)

C = np.matmul(X.T,np.linalg.pinv(Z))
print("Dims of C : " + str(C.shape))
with np.printoptions(precision=4, suppress=True, formatter={'float': '{:0.4f}'.format}, linewidth=200):
    print("Computed C: ")
    print(C)

#U = u_star[:,:-1,:].reshape(M*trajectory_length,-1)
U1 = u_star[:,:-1,0].reshape(1,-1)
U2 = u_star[:,:-1,1].reshape(1,-1)

bi_U1 = Z * np.ones((len(monomials),1))*U1
bi_U2 = Z * np.ones((len(monomials),1))*U2
bi_U = np.vstack((bi_U1, bi_U2))

Y = x_star[:,1:,:].reshape(M*(trajectory_length-1),-1)
Z_prime = basis_fun(Y)

AB = np.matmul(Z_prime,np.linalg.pinv(np.vstack((Z, bi_U))))
A = AB[:,:len(monomials)]
B = AB[:,len(monomials):]

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
X_full = x_star.reshape(M*(trajectory_length),-1)
Z_full = basis_fun(X_full).T.reshape(M,trajectory_length,-1)
Z = Z_full
U = u_star

print("Dims of Z : " + str(Z.shape))
print("Dims of U : " + str(U.shape))

ctrb = np.hstack(
        [B] + [np.linalg.matrix_power(A, i) @ B
                  for i in range(1, A.shape[0])])
print("Dims of controllability matrix: " + str(ctrb.shape))
print("Rank of controllability matrix: " + str(np.linalg.matrix_rank(ctrb)))

Q_re = iBiPMP(A, B, Z, U)

print("Computed Q: ")
with np.printoptions(precision=4, suppress=True, formatter={'float': '{:0.4f}'.format}, linewidth=200):
    print(Q_re)


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

# Lifted State dynamics
f = C @ A @ ca_basis + C @ B[:,:len(monomials)] @ ca_basis * u1 + C @ B[:,len(monomials):] @ ca_basis * u2
f_x = ca.jacobian(f, ca.vertcat(x1,x2,x3))
f_u = ca.jacobian(f, ca.vertcat(u1,u2))

# Cost
R = 0.1*np.identity(2)
phi = ca_basis.T @ Q_re @ ca_basis + ca.vertcat(u1,u2).T @ R @ ca.vertcat(u1,u2)
phi_x = ca.jacobian(phi, ca.vertcat(x1, x2, x3))
phi_u = ca.jacobian(phi, ca.vertcat(u1, u2))

# Compute optimal trajectory
traj_x1,traj_x2,_,_,_ = generateTrajectory(test_start_pt, trajectory_length, 1, 
    ca.Function('phi', [x1, x2, x3, u1, u2], [phi]), ca.Function('f', [x1, x2, x3, u1, u2], [f]))

# Plot LQR trajectory in original states (x,u)
plt.plot(traj_x1, traj_x2, 'g-o', label='Bilinear trajectory')
# Plot the actual trajectory for the test set
plt.plot(test_states[:, 0], test_states[:, 1], 'k-x', label='Actual trajectory')
plt.legend(loc='lower right')
plt.grid()
plt.show()