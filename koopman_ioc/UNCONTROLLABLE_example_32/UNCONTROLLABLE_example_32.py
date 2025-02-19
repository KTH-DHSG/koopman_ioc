import numpy as np
import matplotlib.pyplot as plt
import casadi as ca

from nIOC import *


# Supress scientific notation when printing NumPy arrays
np.set_printoptions(precision=5,suppress=True)


## ---------------------------- FORWARD NONLINEAR OPTIMAL CONTROL ----------------------------

# Symbols/expressions
x1 = ca.MX.sym('x1')
x2 = ca.MX.sym('x2')
u = ca.MX.sym('u')

a = 0.5
b = 0.5

# State dynamics
f = ca.vertcat(a*x1, b*x2 + (b-a**2)*x1**2 + u)
f_x = ca.jacobian(f, ca.vertcat(x1,x2))
f_u = ca.jacobian(f, u)

# Cost
phi = ca.vertcat(x1**2, (x2+x1**2)**2, x1**4)
phi_x = ca.jacobian(phi, ca.vertcat(x1, x2))
phi_u = ca.jacobian(phi, u)


## FORWARD NONLINEAR OPTIMAL CONTROL

## Forward optimal control problem

# Weights to be learned
weights = np.array([1/2, 1, 1])

# Maximum discrete trajectory points
trajectory_length = 10

# Number of trajectories to generate (training set) + 1 (test set)
Mt = 5

# Optimal LQR state and input
x_star = np.zeros((Mt,trajectory_length,2))
u_star = np.zeros((Mt,trajectory_length,1))

start_pt = np.zeros((Mt,2))

for i in range(Mt):
    start_pt[i,:] = np.matrix(10*np.random.rand(2)-5)
    traj_x1,traj_x2,traj_u = generateTrajectory(start_pt[i,:], trajectory_length, weights, 
        ca.Function('phi', [x1, x2, u], [phi]), ca.Function('f', [x1, x2, u], [f]))
    
    x_star[i,:,0] = traj_x1[:-1].reshape(trajectory_length)
    x_star[i,:,1] = traj_x2[:-1].reshape(trajectory_length)
    u_star[i,:,:] = traj_u

s = x_star
c = u_star

# Plot the actual trajectory for the test set
plt.plot(traj_x1, traj_x2, 'b-o', label='Actual trajectory')
plt.show()

## ---------------------------- INVERSE NONLINEAR OPTIMAL CONTROL ----------------------------

traj_len = (trajectory_length+1)
states = s[-1,:traj_len,:]
controls = c[-1,:traj_len,:]

M_ = M(states, ca.Function('phi_x', [x1, x2], [phi_x]))
N_ = N(controls, ca.Function('phi_u', [u], [phi_u]))
A_ = A(states,controls, ca.Function('f_x', [x1, x2, u], [f_x]))
B_ = B(states,controls, ca.Function('f_u', [x1, x2, u], [f_u]))
V_ = V(states,controls, ca.Function('f_x', [x1, x2, u], [f_x]))

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
weights_normalized = weights / np.linalg.norm(weights)
weights_normalized_bias = weights_normalized * sol['x'][0] / weights_normalized[0]

traj_x, traj_y, traj_u = generateTrajectory(start_pt[Mt-1,:], trajectory_length, weights_learned, 
        ca.Function('phi', [x1, x2, u], [phi]), ca.Function('f', [x1, x2, u], [f]))

# Plot the estimated trajectory for the test set
plt.plot(traj_x, traj_y, 'k-o', label='nIOC Estimated trajectory')


## Extra data to assess solution
# Add a trajectory from a random Q to the plot of the test set
weights_random = np.random.rand(*weights.shape)
traj_x_random, traj_y_random, _ = generateTrajectory(start_pt[Mt-1,:], 
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




## ---------------------------- INVERSE LQR ----------------------------




## ---------------------------- Forward LQR to ORIGINAL STATES ----------------------------

