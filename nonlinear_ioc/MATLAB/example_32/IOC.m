%% Centralized forward OC & IOC problem
clear; clc; close all;

import casadi.*

syms x1 x2 u real;

a = 0.5;
b = 0.5;

% State dynamics
f = [a*x1; b*x2+(b-a^2)*x1^2 + u];
f_x = matlabFunction(jacobian(f,[x1,x2]));
f_u = matlabFunction(jacobian(f,u));

% Cost
end_pt = [0,0];
phi = [x1^2; (x2+x1^2)^2; x1^4];
phi_x = matlabFunction(jacobian(phi,[x1,x2]));
phi_u = matlabFunction(jacobian(phi,u));


%% Forward optimal control problem

% Weights to be learned
weights = [1/2, 1, 1];

% Maximum discrete trajectory points
trajectory_length = 40;

[traj_x,traj_y,traj_u] = generateTrajectory([10 10], ...
    trajectory_length, weights, matlabFunction(phi), matlabFunction(f));
s = [traj_x(:),traj_y(:)]; 
c = [traj_u(:)];

plot(traj_x(:),traj_y(:),'b-o', 'DisplayName', 'Actual trajectory'); 
hold on;
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
g = vertcat(z(:),sum(z));
lb = [0*ones(size(phi,1),1);1]; 
ub = [inf*ones(size(phi,1),1);1];

prob = struct('f', 1/2*(z'*(R'*R)*z)^2, 'x', z, 'g', g);
solver = nlpsol('solver', 'ipopt', prob, struct('ipopt', ...
    struct('max_iter',5000)));
sol = solver('x0', 0, 'lbg', lb, 'ubg', ub);

% Generate trajectory using the learned weights
[traj_x,traj_y,traj_u] = generateTrajectory([10 10], ...
    trajectory_length, full(sol.x)', matlabFunction(phi), ...
    matlabFunction(f));

plot(traj_x(:),traj_y(:),'k-o','Displayname','Estimated trajectory');


%% Extra data to assess solution

% Add a random trajectory to the plot to check the results are not trivial
[traj_x,traj_y,~] = generateTrajectory([10 10], trajectory_length, ...
    rand(size(weights)), matlabFunction(phi), matlabFunction(f));
plot(traj_x(:),traj_y(:),'g-o','Displayname','Random trajectory');

% Show original normalized weights and the learned weights
normalized_weights = weights/norm(weights);
disp("Learned weights | Original normalized weights")
disp([full(sol.x),normalized_weights'])

% Show original normalized weights (without bias) and the learned weights
disp("Learned weights | Original normalized weights (without bias)")
disp([full(sol.x),normalized_weights'*sol.x(1)/normalized_weights(1)])
