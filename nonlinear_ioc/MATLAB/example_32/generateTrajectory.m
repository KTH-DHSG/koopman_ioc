function [x,y,u] = generateTrajectory(start_pt,T,z,phi,f)
    import casadi.*
    % Declare casadi decision variables
    u = SX.sym('u',T);

    % Dynamics and control cost
    dynamics = @(u,px,py) f(u,px,py);  
    px=start_pt(1);py=start_pt(2);
    traj_x = []; traj_y = [];
    cost_k = 0;
    for k = 1:T
        next = dynamics(u(k),px,py);
        px = next(1); py = next(2);
        traj_x = [traj_x,px]; traj_y = [traj_y,py];
        phi_k = phi(px,py);
        cost_k = cost_k + z*phi_k;
    end
     
    cost_x_T = 0; %(px-end_pt(1))^2 + (py-end_pt(2))^2;
    f = (1/T)*cost_k  + cost_x_T;

    % Casadi decision variables, constraints and objective func
    u_max = inf;
    g = vertcat(u(:),traj_x(:),traj_y(:));
    lb = [-u_max*ones(T,1);-1*ones(T,1);-1*ones(T,1)]; 
    ub = [u_max*ones(T,1);1*ones(T,1);1*ones(T,1)];

    prob = struct('f', f, 'x', u, 'g', g);
    solver = nlpsol('solver', 'ipopt', prob, struct('ipopt',struct('max_iter',1000)));
    
    % Get the optimal solution and plot
    sol = solver('x0',0,'lbg', lb, 'ubg', ub);
    u_sol = full(sol.x(1:T)); v_sol = full(sol.x(T+1:end));
    
    px=start_pt(1);py=start_pt(2);
    px_sol = [px]; py_sol = [py];
    for k = 1:T
        next = dynamics(u_sol(k),px,py);
        px = next(1); py = next(2);
        px_sol = [px_sol, px]; py_sol = [py_sol, py]; 
    end
    
    x = px_sol; y=py_sol; u = u_sol;
end