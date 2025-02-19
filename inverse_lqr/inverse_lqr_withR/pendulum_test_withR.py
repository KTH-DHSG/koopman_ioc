import numpy as np
import matplotlib.pyplot as plt

from iLQR_withR import *


# Supress scientific notation when printing NumPy arrays
np.set_printoptions(precision=5,suppress=True)

## ---------------------------- FORWARD LQR ----------------------------

def lqr(A,B,Q,R,N):
    """
    Solve the discrete time lqr controller.
    x[k+1] = A x[k] + B u[k]
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
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

l = .22 # rod length is 2l
m = (2*l)*(.006**2)*(3.14/4)*7856 # rod 6 mm diameter, 44cm length, 7856 kg/m^3
M = .4
dt = .02 # 20 ms
g = 9.8

A = np.matrix([[1, dt, 0, 0],[0,1, -(3*m*g*dt)/(7*M+4*m),0],[0,0,1,dt],[0,0,(3*g*(m+M)*dt)/(l*(7*M+4*m)),1]])
B = np.matrix([[0],[7*dt/(7*M+4*m)],[0],[-3*dt/(l*(7*M+4*m))]])

ctrb = np.hstack(
        [B] + [np.linalg.matrix_power(A, i) @ B
                  for i in range(1, A.shape[0])])
print("Dims of controllability matrix: " + str(ctrb.shape))
print("Rank of controllability matrix: " + str(np.linalg.matrix_rank(ctrb)))

#print(A, B)

Q = np.matrix("1 0 0 0; 0 .0001 0 0 ; 0 0 1 0; 0 0 0 .0001")
R = np.matrix("1")

nsteps = 150
K = lqr(A,B,Q,R, nsteps)
print(K[0].shape)

# Optimal LQR state and input
X_star = []; U_star = []
M = 5
for m in range(M):
  x_star = np.zeros((nsteps,4))
  u_star = np.zeros((nsteps-1,1))
  xk = np.matrix(2*np.random.rand(4,1)-1)
  x_star[0,:] = xk.ravel()
  for i in range(len(K)):
      uk = K[i]*xk
      xk = A*xk + B*uk
      x_star[i+1,:] = xk.ravel()
      u_star[i,:] = uk
  X_star.append(x_star); U_star.append(u_star)

X_star = np.array(X_star)
U_star = np.array(U_star)

print("Dims of X* : " + str(X_star.shape))
print("Dims of U* : " + str(U_star.shape))

# Check that there is at least n linearly independent elements in the second to last place in X*
#print("Rank of the second to last element in X* : " + str(np.linalg.matrix_rank(X_star[:,-2,:].T))) #This does not work because the tolerance is too small
U, s, V = np.linalg.svd(X_star[:,-2,:])
TOLERANCE = 5e-8
print("Number of linearly independent second to last elements in X* : " + str(np.sum(s > TOLERANCE)))

# Inverse LQR
Q_re, R_re = iLQR_withR(A, B, X_star, U_star)

print("Computed Q: ")
print(Q_re)
print("Original Q: ")
print(Q)

print("Computed R: ")
print(R_re)
print("Original R: ")
print(R)


## Plot to check equivalence
time = np.linspace(0, 2, nsteps-1, endpoint=True)
plt.plot(time, X_star[0,:-1,:], label="original state")
plt.plot(time, U_star[0,:,:], label='original control')

# Re compute trajectory
K_re = lqr(A,B,Q_re,R_re, nsteps)
re_x_star = np.zeros((nsteps,4))
re_u_star = np.zeros((nsteps-1,1))
xk = np.matrix(X_star[0,0,:]).T
re_x_star[0,:] = xk.ravel()
for i in range(len(K_re)):
    uk = K_re[i]*xk
    xk = A*xk + B*uk
    re_x_star[i+1,:] = xk.ravel()
    re_u_star[i,:] = uk

plt.plot(time, re_x_star[:-1,:], label="re-comp state")
plt.plot(time, re_u_star, label='re-comp control')

plt.legend()
plt.grid()
plt.show()