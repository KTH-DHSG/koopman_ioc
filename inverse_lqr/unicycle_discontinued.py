import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from tqdm import tqdm
from scipy.optimize import minimize, LinearConstraint
 

### THIS CODE DOES NOT WORK BECAUSE iLQR DOES NOT WORK FOR TIME-VARYING A & B ###


## ---------------------------- FORWARD LQR ----------------------------

# Modified code from:
# Author: Addison Sears-Collins
# https://automaticaddison.com/linear-quadratic-regulator-lqr-with-python-code-example/
# Description: Linear Quadratic Regulator example 
#   (two-wheeled differential drive robot car)
 
######################## DEFINE CONSTANTS #####################################
# Supress scientific notation when printing NumPy arrays
np.set_printoptions(precision=3,suppress=True)
 
# Maximum time of the simulation
sim_time = 5 # seconds
# Time interval
dt = 0.2 # seconds
# Rounded up max time size
N = int(-(-sim_time//dt))

# Number of optimal trajectories to generate
M = 5

# Optimal LQR state and input
x_star = np.zeros((N,3,M))
u_star = np.zeros((N,2,M))

# Speed limits
max_linear_velocity = 1.0 # meters per second
max_angular_velocity = 0.5708 # radians per second
 
def getB(yaw, deltat):
    """
    Calculates and returns the B matrix
    3x2 matix ---> number of states x number of control inputs
 
    Expresses how the state of the system [x,y,yaw] changes
    from t-1 to t due to the control commands (i.e. control inputs).
     
    :param yaw: The yaw angle (rotation angle around the z axis) in radians 
    :param deltat: The change in time from timestep t-1 to t in seconds
     
    :return: B matrix ---> 3x2 NumPy array
    """
    B = np.array([  [np.cos(yaw)*deltat, 0],
                    [np.sin(yaw)*deltat, 0],
                    [0, deltat]])
    return B
 
 
def state_space_model(A, state_t_minus_1, B, control_input_t_minus_1):
    """
    Calculates the state at time t given the state at time t-1 and
    the control inputs applied at time t-1
     
    :param: A   The A state transition matrix
        3x3 NumPy Array
    :param: state_t_minus_1     The state at time t-1  
        3x1 NumPy Array given the state is [x,y,yaw angle] ---> 
        [meters, meters, radians]
    :param: B   The B state transition matrix
        3x2 NumPy Array
    :param: control_input_t_minus_1     Optimal control inputs at time t-1  
        2x1 NumPy Array given the control input vector is 
        [linear velocity of the car, angular velocity of the car]
        [meters per second, radians per second]
         
    :return: State estimate at time t
        3x1 NumPy Array given the state is [x,y,yaw angle] --->
        [meters, meters, radians]
    """
    # These next 6 lines of code which place limits on the angular and linear 
    # velocities of the robot car can be removed if you desire.
    #control_input_t_minus_1[0] = np.clip(control_input_t_minus_1[0],
    #                                                                        -max_linear_velocity,
    #                                                                        max_linear_velocity)
    #control_input_t_minus_1[1] = np.clip(control_input_t_minus_1[1],
    #                                                                        -max_angular_velocity,
    #                                                                        max_angular_velocity)
    state_estimate_t = (A @ state_t_minus_1) + (B @ control_input_t_minus_1) 
             
    return state_estimate_t
     
def lqr(Q, R, A, B, N):
    """
    Discrete-time linear quadratic regulator for a time-varying system (non-linear).
 
    Compute the optimal control inputs given a nonlinear system, cost matrices, 
    current state, and a final state.
     
    Compute the control variable K that minimizes the cumulative cost.
 
    Solve for P for iteration i.
 
    :param Q: The state cost matrix
        3x3 NumPy Array
    :param R: The input cost matrix
        2x2 NumPy Array
    :param: A   The A state transition matrix
        3x3 NumPy Array
    :param: B   The B state transition matrix
        3x2 NumPy Array
    :param: N  Trajectory discrete length
 
    :return: K: Optimal gain for the current state 
        2x3 NumPy Array
    """
 
    P = [None]*N
    P[N-1] = np.zeros(A.shape)
    # For i = N, ..., 1
    for i in range(N-1, 0, -1):
        # Discrete-time Algebraic Riccati equation to calculate the optimal 
        # state cost matrix
        P[i-1] = Q + A.T @ P[i] @ A - (A.T @ P[i] @ B) @ np.linalg.pinv(
            R + B.T @ P[i] @ B) @ (B.T @ P[i] @ A)      
 
    # Create a list of N elements
    K = [None]*(N-1)
 
    # For i = 0, ..., N - 1
    for i in range(N-1):
        # Calculate the optimal feedback gain K
        K[i] = -np.linalg.pinv(R + B.T @ P[i+1] @ B) @ B.T @ P[i+1] @ A
    
    return K
 
######################## MAIN PROGRAM #####################################

# Basis actual state
# Our robot starts out at the origin (x=0 meters, y=0 meters), and 
# the yaw angle is 0 radians. 
basis_actual_state_x = np.array([0,0,0]) 

# Basis desired state [x,y,yaw angle]
# [meters, meters, radians]
basis_desired_state_xf = np.array([2.000,2.000,np.pi/2])  
    
# A matrix
# 3x3 matrix -> number of states x number of states matrix
# Expresses how the state of the system [x,y,yaw] changes 
# from t-1 to t when no control command is executed.
# Typically a robot on wheels only drives when the wheels are told to turn.
# For this case, A is the identity matrix.
# Note: A is sometimes F in the literature.
A = np.array([  [1.0,  0,   0],
                [  0,1.0,   0],
                [  0,  0, 1.0]])

# R matrix
# The control input cost matrix
# Experiment with different R matrices
# This matrix penalizes actuator effort (i.e. rotation of the 
# motors on the wheels that drive the linear velocity and angular velocity).
# The R matrix has the same number of rows as the number of control
# inputs and same number of columns as the number of 
# control inputs.
# This matrix has positive values along the diagonal and 0s elsewhere.
# We can target control inputs where we want low actuator effort 
# by making the corresponding value of R large. 
R = np.array([  [1,   0],  # Penalty for linear velocity effort
                [0,   1]]) # Penalty for angular velocity effort
#R = np.array([[0.01,   0],  # Penalty for linear velocity effort
#            [  0, 0.01]]) # Penalty for angular velocity effort

# Q matrix
# The state cost matrix.
# Experiment with different Q matrices.
# Q helps us weigh the relative importance of each state in the 
# state vector (X, Y, YAW ANGLE). 
# Q is a square matrix that has the same number of rows as 
# there are states.
# Q penalizes bad performance.
# Q has positive values along the diagonal and zeros elsewhere.
# Q enables us to target states where we want low error by making the 
# corresponding value of Q large.
Q = np.array([  [0.639, 0, 0],  # Penalize X position error 
                [0, 1.0, 0],  # Penalize Y position error 
                [0, 0, 1.0]]) # Penalize YAW ANGLE heading error 

# Vector to save the actual randomized last state of each run
last_state = np.zeros((3,M))

# Create M optimal trajectories
for m in tqdm(range(M)):

    # Actual state as a random perturbation of its basis
    actual_state_x = basis_actual_state_x + 0.3*np.random.rand(basis_actual_state_x.shape[0])-0.15

    # Desired state as a random perturbation of its basis
    desired_state_xf = basis_desired_state_xf + 0.3*np.random.rand(basis_desired_state_xf.shape[0])-0.15
    last_state[:,m] = desired_state_xf

    # Initialize solution
    x_star[0,:,m] = actual_state_x

    # Launch the robot, and have it move to the desired goal destination
    for i in range(N-1):
        #print(f'iteration = {i*dt} seconds')
        #print(f'Current State = {actual_state_x}')
        #print(f'Desired State = {desired_state_xf}')
            
        state_error = actual_state_x - desired_state_xf
        state_error_magnitude = np.linalg.norm(state_error)     
        #print(f'State Error Magnitude = {state_error_magnitude}')
            
        B = getB(actual_state_x[2], dt)
            
        # LQR returns the optimal control gain
        K = lqr(Q, R, A, B, N) 
        # We want the system to stabilize at desired_state_xf.
        x_error = actual_state_x - desired_state_xf
        optimal_control_input = K[i] @ x_error
            
        #print(f'Control Input = {optimal_control_input}')
                                        
            
        # We apply the optimal control to the robot
        # so we can get a new actual (estimated) state.
        actual_state_x = state_space_model(A, actual_state_x, B, 
                                        optimal_control_input)  

        x_star[i+1,:,m] = actual_state_x
        u_star[i,:,m] = optimal_control_input

        # Stop as soon as we reach the goal
        # Feel free to change this threshold value.
        """
        if state_error_magnitude < 0.01:
            print("\nGoal Has Been Reached Successfully!")
            x_star[i:,:,m] = actual_state_x
            break
        """
        #print()

# Plot system evolution of the one of the trajectories j
j = M-1
fig, ax = plt.subplots()  # Create a figure and an axes.
ax.axis('on')
ax.plot(x_star[:,0,j], x_star[:,1,j], color='b', label="Robot")
ax.plot(x_star[0,0,j], x_star[0,1,j], 'bo') # Starting point
ax.plot(x_star[-1,0,j], x_star[-1,1,j], 'b+') # Final point
for i in range(N-1):
    plt.arrow(x=x_star[i,0,j], y=x_star[i,1,j], dx=0.5*np.cos(x_star[i,2,j]), 
              dy=0.5*np.sin(x_star[i,2,j]), width=.008, color='c') # Yaw direction
ax.set_xlabel('X')
ax.set_ylabel('Y') 
ax.set_title("System evolution of trajectory "+str(j))
ax.legend(fontsize=13)

plt.show()

# Animate system evolution of the one of the trajectories j
print("Showing animation...")

# Window size
winx = 4
winy = 4

# Robot length and width
l_robot = 0.2
w_robot = 0.3

# Start figure and axes with limits
fig = plt.figure()
ax = plt.axes(xlim=(-1, winx), ylim=(-1, winy))

time_txt = ax.text(0.475, 0.975,'',horizontalalignment='left',verticalalignment='top', transform=ax.transAxes)

cornerX = x_star[0,0,j] + l_robot/2 * np.cos(x_star[0,2,j]) - w_robot/2 * np.sin(x_star[0,2,j])
cornerY = x_star[0,1,j] + l_robot/2 * np.sin(x_star[0,2,j]) + w_robot/2 * np.cos(x_star[0,2,j])

# Add initial point
initial = plt.Rectangle((cornerX, cornerY), l_robot, w_robot, angle=0.0, rotation_point='center', fc='k', alpha=0.3)
plt.gca().add_patch(initial)

shapes = []
shapes.append(plt.Rectangle((cornerX, cornerY), l_robot, w_robot, angle=0.0, rotation_point='center', fc='b'))

def init():
    shapes[0].set_xy((cornerX, cornerY))
    shapes[0].set_angle(np.rad2deg(x_star[0,2,j]))
    ax.add_patch(shapes[0])

    time_txt.set_text('T=0.0 s')

    return shapes + [time_txt,]

def animate(frame):

    cornerX = x_star[frame,0,j] + l_robot/2 * np.cos(x_star[frame,2,j]) - w_robot/2 * np.sin(x_star[frame,2,j])
    cornerY = x_star[frame,1,j] + l_robot/2 * np.sin(x_star[frame,2,j]) + w_robot/2 * np.cos(x_star[frame,2,j])

    shapes[0].set_xy((cornerX, cornerY))
    shapes[0].set_angle(np.rad2deg(x_star[frame,2,j]))
    
    secs = frame*dt
    time_txt.set_text('T=%.1d s' % secs)

    return shapes + [time_txt,]

anim = animation.FuncAnimation(fig, animate, 
                               init_func=init, 
                               frames=N, 
                               interval=dt*1000,
                               blit=True,
                               repeat=False)

plt.show()

#anim.save('animation.mp4', fps=50, extra_args=['-vcodec', 'libx264'])

print("Completed!")


## ---------------------------- INVERSE LQR ---------------------------- 

# Solve minimization problem (12-14) from https://arxiv.org/pdf/1810.12590.pdf

AAA = []
uu = []
for m in range(M-2):
    AA = 0
    for i in range(1,N-1):
        xx = np.array(x_star[i,:,m].T-last_state[:,m].T)

        B = getB(x_star[i,2,m], dt) 

        # Compute "controllability matrix"
        # Save it backwards
        BB = np.vstack(
            [B.T @ np.linalg.matrix_power(A.T, i)
                    for i in range(N-3, 0, -1)] + [B.T])

        # Select a snippet of it for each i
        BB_cut = BB[(BB.shape[0]-2*i):,:]
        BB_i = np.vstack([BB_cut] + [np.zeros((BB.shape[0]-2*i,BB.shape[1]))])

        AA += np.kron(xx,BB_i)

    AAA.append(AA)
    uu.append(u_star[1:N-1,:,m].flatten())

AAA = np.array(AAA)
AAA = AAA.reshape(AAA.shape[0]*AAA.shape[1],AAA.shape[2])
uu = np.array(uu)
uu = uu.reshape(uu.shape[0]*uu.shape[1],1)


# # If we take the full matrix AAA -> no symmetric constraints enforced
# vecQ = np.linalg.lstsq(AAA, -uu, rcond=None)[0]
# Q_re = vecQ.reshape(3, -1, order='F').T

# If we only consider the upper triangular elements of Q -> symmetric constraints enforced
row_indices, col_indices = np.triu_indices(x_star.shape[1])
upper_tri_indices = row_indices * x_star.shape[1] + col_indices
vecQ = np.linalg.lstsq(AAA[:, upper_tri_indices], -uu, rcond=None)[0]
Q_re = np.zeros((x_star.shape[1], x_star.shape[1]))
Q_re[row_indices, col_indices] = vecQ.ravel()
Q_re = Q_re + Q_re.T - np.diag(Q_re.diagonal())

# # If we only consider the diagonal elements of Q -> symmetric & zero off-diagonal constraints enforced
# diagonal_indices = [i * (x_star.shape[1] + 1) for i in range(x_star.shape[1])]
# vecQ = np.linalg.lstsq(AAA[:, diagonal_indices], -uu, rcond=None)[0]
# Q_re = np.zeros((x_star.shape[1], x_star.shape[1]))
# np.fill_diagonal(Q_re, vecQ)


print("Vectorized Q: ")
print(vecQ)

print("Computed Q: ")
print(Q_re)
print("Original Q: ")
print(Q)

"""
OLD METHOD USING DIRECTLY EQ 11

# Initial conditions
Q_0 = np.identity(3)
llambda_0 = np.ones(x_star.shape)
res_0  = np.concatenate((Q_0.flatten(), llambda_0.flatten()))

# Define linear constraints wrt llambda and Q
def constraint_lambda(res):
    res = np.split(res, [9,])
    llambda = res[1].reshape(x_star.shape)
    Q = res[0].reshape(3,3)
    constraint = np.zeros((x_star.shape))
    for i in range(1, N-1):
        constraint[i-1,:] = llambda[i,:] - A_saved.T @ llambda[i+1,:] - Q @ x_star[i,:]
    return constraint.flatten()

def constraint_terminal(res):
    res = np.split(res, [9,])
    llambda = res[1].reshape(x_star.shape)
    Q = res[0].reshape(3,3)
    return llambda[-1,:].flatten()

def constraint_u(res):
    res = np.split(res, [9,])
    llambda = res[1].reshape(x_star.shape)
    Q = res[0].reshape(3,3)
    constraint = np.zeros((u_star.shape))
    for i in range(N-1):
        constraint[i,:] = u_star[i,:] + B_saved[i].T @ llambda[i+1,:]
    return constraint.flatten()

cons = [{'type':'eq', 'fun': constraint_lambda},
        {'type':'eq', 'fun': constraint_u},
        {'type':'eq', 'fun': constraint_terminal}]

# Define objective function
def objective_function(res):
    return 0

# Construct the problem
sol = minimize(
    objective_function,
    x0=res_0,
    constraints=cons,
)

print(sol)

sol = np.split(sol.x, [9,])
llambda = sol[1].reshape(x_star.shape)
Q = sol[0].reshape(3,3)
print(llambda)
print(Q)
"""


## ---------------------------- FORWARD LQR WITH INVERSE DATA ----------------------------
    
m = j

# Actual state as a random perturbation of its basis
actual_state_x = x_star[0,:,m]

# Desired state as a random perturbation of its basis
desired_state_xf = last_state[:,m]

# Launch the robot, and have it move to the desired goal destination
for i in range(N-1):
    #print(f'iteration = {i*dt} seconds')
    #print(f'Current State = {actual_state_x}')
    #print(f'Desired State = {desired_state_xf}')
        
    state_error = actual_state_x - desired_state_xf
    state_error_magnitude = np.linalg.norm(state_error)     
    #print(f'State Error Magnitude = {state_error_magnitude}')
        
    B = getB(actual_state_x[2], dt)
        
    # LQR returns the optimal control input
    K = lqr(Q_re, R, A, B, N) 
    # We want the system to stabilize at desired_state_xf.
    x_error = actual_state_x - desired_state_xf
    optimal_control_input = K[i] @ x_error
        
    #print(f'Control Input = {optimal_control_input}')
                                    
        
    # We apply the optimal control to the robot
    # so we can get a new actual (estimated) state.
    actual_state_x = state_space_model(A, actual_state_x, B, 
                                    optimal_control_input)  

    x_star[i,:,m] = actual_state_x
    u_star[i,:,m] = optimal_control_input

    # Stop as soon as we reach the goal
    # Feel free to change this threshold value.
    if state_error_magnitude < 0.01:
        print("\nGoal Has Been Reached Successfully!")
        x_star[i:,:,m] = actual_state_x
        break
            
    #print()

# Plot system evolution of the one of the trajectories j
j = M-1
fig, ax = plt.subplots()  # Create a figure and an axes.
ax.axis('on')
ax.plot(x_star[:,0,j], x_star[:,1,j], color='b', label="Robot")
ax.plot(x_star[0,0,j], x_star[0,1,j], 'bo') # Starting point
ax.plot(x_star[-1,0,j], x_star[-1,1,j], 'b+') # Final point
for i in range(N-1):
    plt.arrow(x=x_star[i,0,j], y=x_star[i,1,j], dx=0.5*np.cos(x_star[i,2,j]), 
              dy=0.5*np.sin(x_star[i,2,j]), width=.008, color='c') # Yaw direction
ax.set_xlabel('X')
ax.set_ylabel('Y') 
ax.set_title("System evolution of recreated trajectory "+str(j))
ax.legend(fontsize=13)

plt.show()

# Animate system evolution of the one of the trajectories j
print("Showing animation...")

# Window size
winx = 4
winy = 4

# Robot length and width
l_robot = 0.2
w_robot = 0.3

# Start figure and axes with limits
fig = plt.figure()
ax = plt.axes(xlim=(-1, winx), ylim=(-1, winy))

time_txt = ax.text(0.475, 0.975,'',horizontalalignment='left',verticalalignment='top', transform=ax.transAxes)

cornerX = x_star[0,0,j] + l_robot/2 * np.cos(x_star[0,2,j]) - w_robot/2 * np.sin(x_star[0,2,j])
cornerY = x_star[0,1,j] + l_robot/2 * np.sin(x_star[0,2,j]) + w_robot/2 * np.cos(x_star[0,2,j])

# Add initial point
initial = plt.Rectangle((cornerX, cornerY), l_robot, w_robot, angle=0.0, rotation_point='center', fc='k', alpha=0.3)
plt.gca().add_patch(initial)

shapes = []
shapes.append(plt.Rectangle((cornerX, cornerY), l_robot, w_robot, angle=0.0, rotation_point='center', fc='b'))

def init():
    shapes[0].set_xy((cornerX, cornerY))
    shapes[0].set_angle(np.rad2deg(x_star[0,2,j]))
    ax.add_patch(shapes[0])

    time_txt.set_text('T=0.0 s')

    return shapes + [time_txt,]

def animate(frame):

    cornerX = x_star[frame,0,j] + l_robot/2 * np.cos(x_star[frame,2,j]) - w_robot/2 * np.sin(x_star[frame,2,j])
    cornerY = x_star[frame,1,j] + l_robot/2 * np.sin(x_star[frame,2,j]) + w_robot/2 * np.cos(x_star[frame,2,j])

    shapes[0].set_xy((cornerX, cornerY))
    shapes[0].set_angle(np.rad2deg(x_star[frame,2,j]))
    
    secs = frame*dt
    time_txt.set_text('T=%.1d s' % secs)

    return shapes + [time_txt,]

anim = animation.FuncAnimation(fig, animate, 
                               init_func=init, 
                               frames=N, 
                               interval=dt*1000,
                               blit=True,
                               repeat=False)

plt.show()

#anim.save('animation.mp4', fps=50, extra_args=['-vcodec', 'libx264'])

print("Completed!")