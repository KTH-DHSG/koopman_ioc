import numpy as np
import matplotlib.pyplot as plt
import casadi as ca

from nIOC import *


# Supress scientific notation when printing NumPy arrays
np.set_printoptions(precision=5,suppress=True)

# Symbols/expressions
x = ca.MX.sym('x')
y = ca.MX.sym('y')
theta = ca.MX.sym('theta')
w = ca.MX.sym('w')
v = ca.MX.sym('v')

lambda_x = ca.MX.sym('lambda_x')
lambda_y = ca.MX.sym('lambda_y')
lambda_th = ca.MX.sym('lambda_th')

# State dynamics
f = ca.vertcat(x + v*ca.cos(theta), y + v*ca.sin(theta), w + theta)
f_x = ca.jacobian(f, ca.vertcat(x,y,theta))
f_u = ca.jacobian(f, ca.vertcat(w,v))

# Cost
obstacles = np.vstack((np.zeros((1, 11)), np.arange(0, -1.1, -0.1))).T
#obstacles = np.vstack((np.zeros((1, 1)), - 0.5)).T
end_pt = [1,0]
phi = ca.vertcat(v**2, w**2, (x - end_pt[0])**2, (y - end_pt[1])**2)
for i in range(obstacles.shape[0]):
    phi = ca.vertcat(phi, ca.exp(-10 * ((x - obstacles[i, 0])**2 + (y - obstacles[i, 1])**2)))
    # phi = ca.vertcat(phi, ca.exp(-10 * (x - obstacles[i, 0])), ca.exp(-10 * (y - obstacles[i, 1])))

phi_x = ca.jacobian(phi, ca.vertcat(x, y, theta))
phi_u = ca.jacobian(phi, ca.vertcat(w, v))


## Forward optimal control problem

# Weights to be learned
weights = np.array([1e3, 0, 10, 10] + [100] * 1*obstacles.shape[0]) 

# Maximum discrete trajectory points
trajectory_length = 40

traj_x,traj_y,traj_th,traj_v,traj_w = generateTrajectory([-1, -1],
    trajectory_length, weights, ca.Function('phi', [x, y, w, v], [phi]), 
    ca.Function('f', [x, y, theta, w, v], [f]))

s = np.column_stack((traj_x, traj_y, traj_th))
c = np.column_stack((traj_w, traj_v))

# Plot the actual trajectory
plt.plot(traj_x, traj_y, 'b-o', label='Actual trajectory')

# Plot obstacles if they exist
if obstacles is not None and not np.all(obstacles == 0):
    plt.plot(obstacles[:, 0], obstacles[:, 1], '*r', label='Obstacles')


## IOC
traj_len = 40
states = s[:traj_len,:]
controls = c[:traj_len,:]

M_ = M(states, ca.Function('phi_x', [x, y], [phi_x]))
N_ = N(controls, ca.Function('phi_u', [w, v], [phi_u]))
A_ = A(states,controls, ca.Function('f_x', [theta, v], [f_x]))
B_ = B(states,controls, ca.Function('f_u', [theta], [f_u]))
V_ = V(states,controls, ca.Function('f_x', [theta, v], [f_x]))

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
solver_opts = {'ipopt': {'max_iter': 5000}}
solver = ca.nlpsol('solver', 'ipopt', prob, solver_opts)

# Solve the optimization problem
sol = solver(x0=0, lbg=lb, ubg=ub)

# Generate trajectory using the learned weights
weights_learned = sol['x']
weights_normalized = weights / np.linalg.norm(weights)
weights_normalized_bias = weights_normalized * sol['x'][0] / weights_normalized[0]

traj_x, traj_y, traj_th, traj_v, traj_w = generateTrajectory([-1, -1], 
    trajectory_length, weights_learned, ca.Function('phi', [x, y, w, v], [phi]), 
    ca.Function('f', [x, y, theta, w, v], [f]))

# Plot the estimated trajectory
plt.plot(traj_x, traj_y, 'k-o', label='Estimated trajectory')


## Extra data to assess solution

# Add a trajectory from a random Q to the plot
weights_random = np.random.rand(*weights.shape)
traj_x_random, traj_y_random, _, _, _ = generateTrajectory([-1, -1], 
    trajectory_length, weights_random, ca.Function('phi', [x, y, w, v], [phi]), 
    ca.Function('f', [x, y, theta, w, v], [f]))

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