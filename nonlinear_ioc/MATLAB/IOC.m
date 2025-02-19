%% Centralized forward OC & IOC problem
clear; clc; close all;

import casadi.*

syms x y theta w v real;
syms lambda_x lambda_y lambda_th real;

% State dynamics
f = [x + v*cos(theta); y + v*sin(theta); w + theta];
f_x = matlabFunction(jacobian(f,[x,y,theta]));
f_u = matlabFunction(jacobian(f,[v,w]));

% Cost
obstacles = [zeros(1,11); 0:-0.1:-1]';
%obstacles = [zeros(1,1); -0.5]';
end_pt = [1,0];
phi = [v^2; w^2; (x-end_pt(1))^2; (y-end_pt(2))^2];
for i = 1:size(obstacles,1)
    phi = [phi; exp(-10*((x-obstacles(i,1))^2 + (y - obstacles(i,2))^2))];
end

phi_x = matlabFunction(jacobian(phi,[x,y,theta]));
phi_u = matlabFunction(jacobian(phi,[v,w]));


%% Forward optimal control problem

% Weights to be learned
weights = [1e3, 0, 10, 10, 100*ones(1,size(obstacles,1))];

% Maximum discrete trajectory points
trajectory_length = 40;

[traj_x,traj_y,traj_th,traj_v,traj_w] = generateTrajectory([-1 -1], ...
    trajectory_length, weights, matlabFunction(phi), matlabFunction(f));
s = [traj_x(:),traj_y(:),traj_th(:)]; 
c = [traj_v(:),traj_w(:)];

plot(traj_x(:),traj_y(:),'b-o', 'DisplayName', 'Actual trajectory'); 
hold on;
if(~isempty(obstacles))
    plot(obstacles(:,1),obstacles(:,2),'*r','DisplayName','Obstacles');
end
legend('Location','southeast');


%% IOC
traj_len = 40;
states = s(1:traj_len,:);
controls = c(1:traj_len,:);

M_ = M(states,phi_x);
N_ = N(controls,phi_u);
A_ = A(states,controls,f_x);
B_ = B(states,controls,f_u);
V_ = V(states,controls,f_x);

F = B_*inv(A_)*M_ + N_; E = B_*inv(A_)*V_;
 if(rank(E)==size(states,2))
     R = F - E*inv(E'*E)*E'*F;
 else
     disp("Not IOC effective")
 end

% Solve centralized, single-agent problem
z = SX.sym('z',size(phi,1));
g = z(1);
lb = 1; 
ub = 1;

prob = struct('f', 1/2*(z'*(R'*R)*z)^2, 'x', z, 'g', g);
solver = nlpsol('solver', 'ipopt', prob, struct('ipopt', ...
    struct('max_iter',1000)));
sol = solver('x0', 0, 'lbg', lb, 'ubg', ub);

% Generate trajectory using the learned weights
[traj_x,traj_y,traj_th,traj_v,traj_w] = generateTrajectory([-1 -1], ...
    trajectory_length, full(sol.x)', matlabFunction(phi), ...
    matlabFunction(f));

plot(traj_x(:),traj_y(:),'k-o','Displayname','Estimated trajectory');


%% Extra data to assess solution

% Add a random trajectory to the plot to check the results are not trivial
[traj_x,traj_y,~,~,~] = generateTrajectory([-1 -1], trajectory_length, ...
    rand(size(weights)), matlabFunction(phi), matlabFunction(f));
plot(traj_x(:),traj_y(:),'g-o','Displayname','Random trajectory');

% Show original normalized weights and the learned weights
normalized_weights = weights/norm(weights);
disp("Learned weights | Original normalized weights")
disp([full(sol.x),normalized_weights'])

% Show original normalized weights (without bias) and the learned weights
disp("Learned weights | Original normalized weights (without bias)")
disp([full(sol.x),normalized_weights'*sol.x(1)/normalized_weights(1)])
