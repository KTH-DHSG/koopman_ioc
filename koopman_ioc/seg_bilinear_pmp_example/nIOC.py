import casadi as ca
import numpy as np


def generateTrajectory(start_pt, T, z, phi, f):
    # Declare CasADi decision variables
    u1 = ca.SX.sym('u1', T)
    u2 = ca.SX.sym('u2', T)

    # Dynamics and control cost
    px, py = start_pt[0], start_pt[1]
    traj_x = []
    traj_y = []
    cost_k = 0

    for k in range(T):
        next = f(px, py, u1[k], u2[k])
        px = next[0]
        py = next[1]
        traj_x.append(px)
        traj_y.append(py)
        phi_k = phi(px, py, u1[k], u2[k])
        cost_k += ca.dot(z, phi_k)

    cost_x_T = 0  # (px - end_pt[0])**2 + (py - end_pt[1])**2
    f_cost = (1 / T) * cost_k + cost_x_T

    # CasADi decision variables, constraints, and objective function
    u = ca.vertcat(u1,u2)
    u1_max = np.inf
    u2_max = np.inf
    traj_x_concat = ca.vertcat(*traj_x)
    traj_y_concat = ca.vertcat(*traj_y)
    g = ca.vertcat(u)
    lb = ca.vertcat([-u1_max] * T, [-u2_max] * T)
    ub = ca.vertcat([u1_max] * T, [u2_max] * T)

    prob = {'f': f_cost, 'x': u, 'g': g}
    solver_opts = {'ipopt': {'max_iter': 1000}, 'ipopt.print_level':0, 'print_time':0}
    solver = ca.nlpsol('solver', 'ipopt', prob, solver_opts)

    # Get the optimal solution
    sol = solver(x0=0, lbg=lb, ubg=ub)
    u1_sol = sol['x'][:T]
    u2_sol = sol['x'][T:]

    px, py = start_pt[0], start_pt[1]
    px_sol = np.zeros((u1_sol.shape[0]+1,1))
    px_sol[0] = px
    py_sol = np.zeros((u1_sol.shape[0]+1,1))
    py_sol[0] = py

    for k in range(T):
        next = f(px, py, u1_sol[k], u2_sol[k])
        px = next[0]
        py = next[1]
        px_sol[k+1] = px
        py_sol[k+1] = py

    return px_sol, py_sol, np.array(u1_sol), np.array(u2_sol)


def A_fun(states, controls, f_x):
    T, n = states.shape[0], states.shape[1]
    ret = np.eye(n * (T-1))

    for i in range(T - 2):
        ret[n*i:n*(i+1), n*(i+1):n*(i+2)] = -f_x(states[i+1, 0], states[i+1, 1], controls[i+1, 0], controls[i+1, 1]).T

    return ret


def B_fun(states, controls, f_u):
    T, n = states.shape[0], states.shape[1]
    m = controls.shape[1]
    ret = np.zeros((m * (T-1), n * (T-1)))

    for i in range(T-1):
        ret[m*i:m*(i+1), n*i:n*(i+1)] = f_u(states[i, 0], states[i, 1], controls[i, 0], controls[i, 1]).T

    return ret


def M_fun(states, controls, phi_x):
    T = states.shape[0]
    ret = []

    for i in range(1,T):
        ret.append(phi_x(states[i, 0], states[i, 1], controls[i, 0], controls[i, 1]).T)

    return np.vstack(ret)


def N_fun(states, controls, phi_u):
    T = controls.shape[0]
    ret = []

    for i in range(T-1):
        ret.append(phi_u(states[i, 0], states[i, 1], controls[i, 0], controls[i, 1]).T)

    return np.vstack(ret)


def V_fun(states, controls, f_x):
    T, n = states.shape[0], states.shape[1]
    ret = np.zeros((n * (T-1), n))

    ret[-n:, :] = f_x(states[-1, 0], states[-1, 1], controls[-1, 0], controls[-1, 1]).T

    return ret


def generateLiftedTrajectory(start_pt, T, phi, f):
    # Declare CasADi decision variables
    u1 = ca.SX.sym('u1', T)
    u2 = ca.SX.sym('u2', T)

    # Dynamics and control cost
    pz = start_pt
    traj_z = []
    cost_k = 0

    for k in range(T):
        next = f(pz, u1[k], u2[k])
        pz = next
        traj_z.append(pz)
        phi_k = phi(pz, u1[k], u2[k])
        cost_k += phi_k

    cost_x_T = 0  # (px - end_pt[0])**2 + (py - end_pt[1])**2
    f_cost = (1 / T) * cost_k + cost_x_T

    # CasADi decision variables, constraints, and objective function
    u = ca.vertcat(u1,u2)
    u1_max = np.inf
    u2_max = np.inf

    g = ca.vertcat(u)
    lb = ca.vertcat([-u1_max] * T, [-u2_max] * T)
    ub = ca.vertcat([u1_max] * T, [u2_max] * T)

    prob = {'f': f_cost, 'x': u, 'g': g}
    solver_opts = {'ipopt': {'max_iter': 1000}, 'ipopt.print_level':0, 'print_time':0}
    solver = ca.nlpsol('solver', 'ipopt', prob, solver_opts)

    # Get the optimal solution
    sol = solver(x0=0, lbg=lb, ubg=ub)
    u1_sol = sol['x'][:T]
    u2_sol = sol['x'][T:]

    pz = start_pt
    pz_sol = np.zeros((u1_sol.shape[0]+1,4))
    pz_sol[0,:] = pz

    for k in range(T):
        next = f(pz, u1_sol[k], u2_sol[k])
        pz = next
        pz_sol[k+1,:] = np.array(pz).reshape(4,)

    return pz_sol, np.array(u1_sol), np.array(u2_sol)