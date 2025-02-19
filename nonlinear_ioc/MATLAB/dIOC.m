addpath('C:\Users\notan\Downloads\Utilities\casadi-windows-matlabR2016a-v3.5.5')

%% Distributed forward OC & IOC problem
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
%obstacles = [zeros(1,2); 0, -0.5]';
%obstacles = [-0.5 0.5 1; -0.5, -0.5, -0.5]';
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

% Constant parameters
alpha = zeros(1,size(phi,1))';
alpha(1) = 1;

% Number of agents/recording stations
agents = 5;

% Adjacency matrix for the network topology (topology assumed: 1-2-3-4-5)
adj = [0 1 0 0 0;
       1 0 1 0 0;
       0 1 0 1 0;
       0 0 1 0 1;
       0 0 0 1 0];
% Number of neighbors per agent
d = zeros(1, agents);
for i = 1:agents
    d(i) = sum(adj(i,:));
end

% Segment the full trajectory between agents to simulate distributed IOC
start = zeros(agents,1);
last = zeros(agents,1);
agent_seg = round(trajectory_length/agents);
states = cell(1,agents);
controls = cell(1,agents);
for i = 1:agents
    % Obtain the maximum and minimum logical points for each agent
    point_first = max(1, round(agent_seg*(i-1)-agent_seg/2));
    point_last = min(trajectory_length, round(agent_seg*i+agent_seg/2));
    % Get two random points within that range
    first = randi([point_first point_last]);
    second = randi([point_first point_last]);
    % Organize them by value to get starting and finish point
    start(i) = min(first,second);
    last(i) = max(first,second);
    % Obtain states and control trajectory segment for agent i
    states{i} = s(start(i):last(i),:);
    controls{i} = c(start(i):last(i),:);
end

% Compute IOC matrices
G = cell(1,agents);
eff = ones(1,agents);
for i = 1:agents
    M_ = M(states{i},phi_x);
    N_ = N(controls{i},phi_u);
    A_ = A(states{i},controls{i},f_x);
    B_ = B(states{i},controls{i},f_u);
    V_ = V(states{i},controls{i},f_x);
    
    F = B_*inv(A_)*M_ + N_; E = B_*inv(A_)*V_;
     if(rank(E)==size(states{i},2))
         R = F - E*inv(E'*E)*E'*F;
         G{i} = inv((R'*R) + 1/d(i)*(alpha*alpha') + ...
             2*d(i)*eye(size(phi,1)));
     else
         disp("Not IOC effective")
         eff(i) = 0;
         G{i} = 1/d(i);
     end
end

% Initialize distributed parameters
y = rand(size(phi,1),agents);
w_sol = rand(size(phi,1),agents);
mu = rand(1,agents);
v_sol = rand(size(phi,1),agents);

% Use distributed optimization for MAS
max_iter = 50000;
k = 1;
while k < max_iter
    for i = 1:agents
        w_k = w_sol;
        v_k = v_sol;
        y(:,i) = eff(i)*(d(i)*w_k(:,i) - alpha*mu(i) - ...
            d(i)*v_k(:,i) + 1/d(i)*alpha);
        w_sol(:,i) = G{i}*y(:,i) + G{i}*(w_k+v_k)*adj(i,:)';
        mu(i) = eff(i)*(mu(i) + ...
            1/d(i)*alpha'*G{i}*(w_k+v_k)*adj(i,:)' + ...
            1/d(i)*alpha'*G{i}*y(:,i) - 1/d(i));
        v_sol(:,i) = eff(i)*(v_k(:,i) + ...
            G{i}*(w_k+v_k)*adj(i,:)' + G{i}*y(:,i) - ...
            1/d(i)*(w_k)*adj(i,:)');
    end
    k = k + 1;
end

% Generate trajectory using the learned weights (for agent 1)
[traj_x,traj_y,traj_th,traj_v,traj_w] = generateTrajectory([-1 -1], ...
    trajectory_length, w_sol(:,1)', matlabFunction(phi), ...
    matlabFunction(f));

plot(traj_x(:),traj_y(:),'k-o','Displayname','Estimated trajectory');


%% Extra data to assess solution

% Add a random trajectory to the plot to check the results are not trivial
[traj_x,traj_y,~,~,~] = generateTrajectory([-1 -1], trajectory_length, ...
    rand(size(weights)), matlabFunction(phi), matlabFunction(f));
plot(traj_x(:),traj_y(:),'g-o','Displayname','Random trajectory');

% Show original normalized weights and the learned weights (for agent 1)
normalized_weights = weights/norm(weights);
disp("Learned weights | Original normalized weights")
disp([w_sol(:,1),normalized_weights'])

% Show original normalized weights (without bias) and the learned weights
disp("Learned weights | Original normalized weights (without bias)")
disp([w_sol(:,1),normalized_weights'*w_sol(1,1)/normalized_weights(1)])

% Show all the learned weights
disp("Learned weights")
disp(w_sol)

% Show the IOC-effectiveness of the segments
disp("IOC-effectiveness of the segments")
disp(eff)
