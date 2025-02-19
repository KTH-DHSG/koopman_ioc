function [x,y,theta,v,w] = generateTrajectory(start_pt,T,z,phi,f)
    import casadi.*
    % Declare casadi decision variables
    w = SX.sym('w',T);
    v = SX.sym('v',T);

    % Dynamics and control cost
    dynamics = @(px,py,theta,w,v) f(theta,v,w,px,py);  
    px=start_pt(1);py=start_pt(2);theta=pi/4;
    traj_x = []; traj_y = [];
    cost_k = 0;
    for k = 1:T
        next = dynamics(px,py,theta,w(k),v(k));
        px = next(1); py = next(2); theta = next(3);
        traj_x = [traj_x,px]; traj_y = [traj_y,py];
        phi_k = phi(v(k),w(k),px,py);
        cost_k = cost_k + z*phi_k;
    end
     
    cost_x_T = 0; %(px-end_pt(1))^2 + (py-end_pt(2))^2;
    f = (1/T)*cost_k  + cost_x_T;

    % Casadi decision variables, constraints and objective func
    u = vertcat(w,v);
    w_max = inf; v_max = inf;
    g = vertcat(u(:),traj_x(:),traj_y(:));
    lb = [-w_max*ones(T,1);zeros(T,1);-1*ones(T,1);-1*ones(T,1)]; 
    ub = [w_max*ones(T,1);v_max*ones(T,1);1*ones(T,1);1*ones(T,1)];

    prob = struct('f', f, 'x', u, 'g', g);
    solver = nlpsol('solver', 'ipopt', prob, struct('ipopt',struct('max_iter',1000)));
    
    % Get the optimal solution and plot
    sol = solver('x0',0,'lbg', lb, 'ubg', ub);
    w_sol = full(sol.x(1:T)); v_sol = full(sol.x(T+1:end));
    
    px=start_pt(1);py=start_pt(2);theta=pi/4;
    px_sol = [px]; py_sol = [py]; theta_sol = [theta];
    for k = 1:T
        next = dynamics(px,py,theta,w_sol(k),v_sol(k));
        px = next(1); py = next(2); theta = next(3);
        px_sol = [px_sol, px]; py_sol = [py_sol, py]; theta_sol = [theta_sol,theta];
    end
    
    x = px_sol; y=py_sol; theta = theta_sol; v = v_sol; w = w_sol;
end