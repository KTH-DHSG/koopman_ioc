import casadi as ca
import numpy as np


def generateTrajectory(start_pt, T, z, phi, f):
    # Declare CasADi decision variables
    w = ca.SX.sym('w', T)
    v = ca.SX.sym('v', T)

    # Dynamics and control cost
    px, py, theta = start_pt[0], start_pt[1], np.pi / 4
    traj_x = []
    traj_y = []
    cost_k = 0

    for k in range(T):
        next = f(px, py, theta, w[k], v[k])
        px = next[0]
        py = next[1]
        theta = next[2]
        traj_x.append(px)
        traj_y.append(py)
        phi_k = phi(px, py, w[k], v[k])
        cost_k += ca.dot(z, phi_k)

    cost_x_T = 0  # (px - end_pt[0])**2 + (py - end_pt[1])**2
    f_cost = (1 / T) * cost_k + cost_x_T

    # CasADi decision variables, constraints, and objective function
    u = ca.vertcat(w, v)
    w_max = np.inf
    v_max = np.inf
    traj_x_concat = ca.vertcat(*traj_x)
    traj_y_concat = ca.vertcat(*traj_y)
    g = ca.vertcat(u, traj_x_concat, traj_y_concat)
    lb = ca.vertcat([-w_max] * T, [0] * T, [-1] * T, [-1] * T)
    ub = ca.vertcat([w_max] * T, [v_max] * T, [1] * T, [1] * T)

    prob = {'f': f_cost, 'x': u, 'g': g}
    solver_opts = {'ipopt': {'max_iter': 1000}}
    solver = ca.nlpsol('solver', 'ipopt', prob, solver_opts)

    # Get the optimal solution
    sol = solver(x0=0, lbg=lb, ubg=ub)
    w_sol = sol['x'][:T]
    v_sol = sol['x'][T:]

    px, py, theta = start_pt[0], start_pt[1], np.pi / 4
    px_sol = np.zeros((w_sol.shape[0]+1,1))
    px_sol[0] = px
    py_sol = np.zeros((w_sol.shape[0]+1,1))
    py_sol[0] = py
    theta_sol = np.zeros((w_sol.shape[0]+1,1))
    theta_sol[0] = theta

    for k in range(T):
        next = f(px, py, theta, w_sol[k], v_sol[k])
        px = next[0]
        py = next[1]
        theta = next[2]
        px_sol[k+1] = px
        py_sol[k+1] = py
        theta_sol[k+1] = theta

    return px_sol, py_sol, theta_sol, v_sol, w_sol


def A(states, controls, f_x):
    T, n = states.shape[0], states.shape[1]
    ret = np.eye(n * (T-1))

    for i in range(T - 2):
        ret[n*i:n*(i+1), n*(i+1):n*(i+2)] = -f_x(states[i+1, 2], controls[i+1, 1]).T

    return ret


def B(states, controls, f_u):
    T, n = states.shape[0], states.shape[1]
    m = controls.shape[1]
    ret = np.zeros((m * (T-1), n * (T-1)))

    for i in range(T-1):
        ret[m*i:m*(i+1), n*i:n*(i+1)] = f_u(states[i, 2]).T

    return ret


def M(states, phi_x):
    T = states.shape[0]
    ret = []

    for i in range(1,T):
        ret.append(phi_x(states[i, 0], states[i, 1]).T)

    return np.vstack(ret)


def N(controls, phi_u):
    T = controls.shape[0]
    ret = []

    for i in range(T-1):
        ret.append(phi_u(controls[i, 0], controls[i, 1]).T)

    return np.vstack(ret)


def V(states, controls, f_x):
    T, n = states.shape[0], states.shape[1]
    ret = np.zeros((n * (T-1), n))

    ret[-n:, :] = f_x(states[-1, 2], controls[-1, 1]).T

    return ret