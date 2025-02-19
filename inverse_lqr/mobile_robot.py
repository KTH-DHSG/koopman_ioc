import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from tqdm import tqdm
from scipy.optimize import minimize, LinearConstraint

from iLQR import *
 

## ---------------------------- FORWARD LQR ----------------------------
 
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
x_star = np.zeros((M,N,2))
u_star = np.zeros((M,N-1,2))
 
def state_space_model(A, state_t_minus_1, B, control_input_t_minus_1):
    """
    Calculates the state at time t given the state at time t-1 and
    the control inputs applied at time t-1
     
    :param: A   The A state transition matrix
        2x2 NumPy Array
    :param: state_t_minus_1     The state at time t-1  
        2x1 NumPy Array given the state is [x,y,yaw angle] ---> 
        [meters, meters, radians]
    :param: B   The B state transition matrix
        2x2 NumPy Array
    :param: control_input_t_minus_1     Optimal control inputs at time t-1  
        2x1 NumPy Array given the control input vector is 
        [linear velocity of the car, angular velocity of the car]
        [meters per second, radians per second]
         
    :return: State estimate at time t
        2x1 NumPy Array given the state is [x,y,yaw angle] --->
        [meters, meters, radians]
    """

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
basis_actual_state_x = np.array([0,0]) 

# Basis desired state [x,y,yaw angle]
# [meters, meters, radians]
basis_desired_state_xf = np.array([10.000,10.000]) 
    
# A matrix
# 2x2 matrix -> number of states x number of states matrix
# Expresses how the state of the system [x,y] changes 
# from t-1 to t when no control command is executed.
# Typically a robot on wheels only drives when the wheels are told to turn.
# For this case, A is the identity matrix.
# Note: A is sometimes F in the literature.
A = np.array([  [1.0,  0],
                [0,  1.0]])

# B matrix
# 2x2 matrix -> number of states x number of controls matrix
B = np.array([  [1.0, 0],
                [0, 1.0]])

# R matrix
# The control input cost matrix
# Experiment with different R matrices
# This matrix penalizes actuator effort (i.e. rotation of the 
# motors on the wheels that drive the linear velocity).
# The R matrix has the same number of rows as the number of control
# inputs and same number of columns as the number of 
# control inputs.
# This matrix has positive values along the diagonal and 0s elsewhere.
# We can target control inputs where we want low actuator effort 
# by making the corresponding value of R large. 
# [FOR NOW IT CAN ONLY BE THE IDENTITY MATRIX]
R = np.array([  [1,   0],  # Penalty for linear x effort
                [0,   1]]) # Penalty for linear y effort

# Q matrix
# The state cost matrix.
# Experiment with different Q matrices.
# Q helps us weigh the relative importance of each state in the 
# state vector (X, Y). 
# Q is a square matrix that has the same number of rows as 
# there are states.
# Q penalizes bad performance.
# Q has positive values along the diagonal and zeros elsewhere.
# Q enables us to target states where we want low error by making the 
# corresponding value of Q large.
Q = np.array([  [0.639, 0],  # Penalize X position error 
                [0, 1.0]])   # Penalize Y position error

# Vector to save the actual randomized last state of each run
last_state = np.zeros((2,M))

# Create M optimal trajectories
for m in tqdm(range(M)):

    # Actual state as a random perturbation of its basis
    actual_state_x = basis_actual_state_x + 3*np.random.rand(basis_actual_state_x.shape[0])-1.5

    # Desired state as a random perturbation of its basis
    desired_state_xf = basis_desired_state_xf + 3*np.random.rand(basis_desired_state_xf.shape[0])-1.5
    last_state[:,m] = desired_state_xf

    # Initialize solution
    x_star[m,0,:] = actual_state_x - desired_state_xf

    # LQR returns the optimal control gain (computed only once since the params don't change)
    K = lqr(Q, R, A, B, N) 

    # Launch the robot, and have it move to the desired goal destination
    for i in range(N-1):
        #print(f'iteration = {i*dt} seconds')
        #print(f'Current State = {actual_state_x}')
        #print(f'Desired State = {desired_state_xf}')
            
        state_error = actual_state_x - desired_state_xf
        state_error_magnitude = np.linalg.norm(state_error)     
        #print(f'State Error Magnitude = {state_error_magnitude}')
        
        # We want the system to stabilize at desired_state_xf.
        x_error = actual_state_x - desired_state_xf
        optimal_control_input = K[i] @ x_error
            
        #print(f'Control Input = {optimal_control_input}')      
            
        # We apply the optimal control to the robot
        # so we can get a new actual (estimated) state.
        actual_state_x = state_space_model(A, actual_state_x, B, 
                                        optimal_control_input)  

        x_star[m,i+1,:] = actual_state_x - desired_state_xf
        u_star[m,i,:] = optimal_control_input

        # Stop as soon as we reach the goal
        # Feel free to change this threshold value.
        """
        if state_error_magnitude < 0.01:
            print("\nGoal Has Been Reached Successfully!")
            x_star[m,i:,:] = actual_state_x
            break
        """
        #print()

# Plot system evolution of the one of the trajectories j
j = M-1
fig, ax = plt.subplots()  # Create a figure and an axes.
ax.axis('on')
ax.plot(x_star[j,:,0] + last_state[0,j], x_star[j,:,1] + last_state[1,j], color='b', label="Robot")
ax.plot(x_star[j,0,0] + last_state[0,j], x_star[j,0,1] + last_state[1,j], 'bo') # Starting point
ax.plot(x_star[j,-1,0]+last_state[0,j], x_star[j,-1,1]+last_state[1,j], 'b+') # Final point
for i in range(N-1):
   plt.arrow(x=x_star[j,i,0] + last_state[0,j], y=x_star[j,i,1]+ last_state[1,j], dx=0.5*(x_star[j,i+1,0]-x_star[j,i,0]), 
                dy=0.5*(x_star[j,i+1,1]-x_star[j,i,1]), width=.008, color='c') # Yaw direction
ax.set_xlabel('X')
ax.set_ylabel('Y') 
ax.set_title("System evolution of trajectory "+str(j))
ax.legend(fontsize=13)

plt.show()

# Animate system evolution of the one of the trajectories j
print("Showing animation...")

# Window size
winx = basis_desired_state_xf[0]+5
winy = basis_desired_state_xf[1]+5

# Robot length and width
r_robot = (basis_desired_state_xf[0] + basis_desired_state_xf[1])/(2*10*5)

# Start figure and axes with limits
fig = plt.figure()
ax = plt.axes(xlim=(-1, winx), ylim=(-1, winy))

time_txt = ax.text(0.475, 0.975,'',horizontalalignment='left',verticalalignment='top', transform=ax.transAxes)

# Add initial point
initial = plt.Circle((x_star[j,0,0]+last_state[0,j], x_star[j,0,1]+last_state[1,j]), r_robot, fc='k', alpha=0.3)
plt.gca().add_patch(initial)

shapes = []
shapes.append(plt.Circle((x_star[j,0,0]+last_state[0,j], x_star[j,0,1]+last_state[1,j]), r_robot, fc='b'))

def init():
    shapes[0].center = (x_star[j,0,0]+last_state[0,j], x_star[j,0,1]+last_state[1,j])
    ax.add_patch(shapes[0])

    time_txt.set_text('T=0.0 s')

    return shapes + [time_txt,]

def animate(frame):
    shapes[0].center = (x_star[j,frame,0]+last_state[0,j], x_star[j,frame,1]+last_state[1,j])
    
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

print("Dims of X* : " + str(x_star.shape))
print("Dims of U* : " + str(u_star.shape))

# Inverse LQR
Q_re = iLQR(A, B, x_star, u_star)

print("Computed Q: ")
print(Q_re)
print("Original Q: ")
print(Q)


## ---------------------------- FORWARD LQR WITH INVERSE DATA ----------------------------

# Actual state as a random perturbation of its basis
actual_state_x = x_star[j,0,:] + last_state[:,m]

# Desired state as a random perturbation of its basis
desired_state_xf = last_state[:,j]

# LQR returns the optimal control input
K = lqr(Q_re, R, A, B, N) 

# Launch the robot, and have it move to the desired goal destination
for i in range(N-1):
    #print(f'iteration = {i*dt} seconds')
    #print(f'Current State = {actual_state_x}')
    #print(f'Desired State = {desired_state_xf}')
        
    state_error = actual_state_x - desired_state_xf
    state_error_magnitude = np.linalg.norm(state_error)     
    #print(f'State Error Magnitude = {state_error_magnitude}')
        
    # We want the system to stabilize at desired_state_xf.
    x_error = actual_state_x - desired_state_xf
    optimal_control_input = K[i] @ x_error
        
    #print(f'Control Input = {optimal_control_input}')
                                    
        
    # We apply the optimal control to the robot
    # so we can get a new actual (estimated) state.
    actual_state_x = state_space_model(A, actual_state_x, B, 
                                    optimal_control_input)  

    x_star[m,i+1,:] = actual_state_x - desired_state_xf
    u_star[m,i,:] = optimal_control_input

    # Stop as soon as we reach the goal
    # Feel free to change this threshold value.
    """
    if state_error_magnitude < 0.01:
        print("\nGoal Has Been Reached Successfully!")
        x_star[m,i:,:] = actual_state_x
        break
    """
    #print()

# Plot system evolution of the one of the trajectories j
fig, ax = plt.subplots()  # Create a figure and an axes.
ax.axis('on')
ax.plot(x_star[j,:,0]+last_state[0,j], x_star[j,:,1]+last_state[1,j], color='b', label="Robot")
ax.plot(x_star[j,0,0]+last_state[0,j], x_star[j,0,1]+last_state[1,j], 'bo') # Starting point
ax.plot(x_star[j,-1,0]+last_state[0,j], x_star[j,-1,1]+last_state[1,j], 'b+') # Final point
for i in range(N-1):
    plt.arrow(x=x_star[j,i,0]+last_state[0,j], y=x_star[j,i,1]+last_state[1,j], dx=0.5*(x_star[j,i+1,0]-x_star[j,i,0]), 
              dy=0.5*(x_star[j,i+1,1]-x_star[j,i,1]), width=.008, color='c') # Yaw direction
ax.set_xlabel('X')
ax.set_ylabel('Y') 
ax.set_title("System evolution of recreated trajectory "+str(j))
ax.legend(fontsize=13)

plt.show()

# Animate system evolution of the one of the trajectories j
print("Showing animation...")

# Window size
winx = basis_desired_state_xf[0]+5
winy = basis_desired_state_xf[1]+5

# Robot length and width
r_robot = (basis_desired_state_xf[0] + basis_desired_state_xf[1])/(2*10*5)

# Start figure and axes with limits
fig = plt.figure()
ax = plt.axes(xlim=(-1, winx), ylim=(-1, winy))

time_txt = ax.text(0.475, 0.975,'',horizontalalignment='left',verticalalignment='top', transform=ax.transAxes)

# Add initial point
initial = plt.Circle((x_star[j,0,0]+last_state[0,j], x_star[j,0,1]+last_state[1,j]), r_robot, fc='k', alpha=0.3)
plt.gca().add_patch(initial)

shapes = []
shapes.append(plt.Circle((x_star[j,0,0]+last_state[0,j], x_star[j,0,1]+last_state[1,j]), r_robot, fc='b'))

def init():
    shapes[0].center = (x_star[j,0,0]+last_state[0,j], x_star[j,0,1]+last_state[1,j])
    ax.add_patch(shapes[0])

    time_txt.set_text('T=0.0 s')

    return shapes + [time_txt,]

def animate(frame):
    shapes[0].center = (x_star[j,frame,0]+last_state[0,j], x_star[j,frame,1]+last_state[1,j])
    
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
