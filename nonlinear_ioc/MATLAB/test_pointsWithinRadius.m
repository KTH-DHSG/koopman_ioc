% Example usage
s = [1, 2; 3, 4; 5, 6; 7, 8; 9, 10];  % Discrete trajectory
x = [4, 5];                          % Sensor position
Delta = 3;                           % Sensing radius

pointsWithinRadius = pointsWithinSensingRadius(s, x, Delta);

% Display the points within the sensing radius
disp(pointsWithinRadius);