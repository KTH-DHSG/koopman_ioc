function pointsWithinRadius = pointsWithinSensingRadius(s, x, Delta)
    pointsWithinRadius = [];
    
    for i = 1:size(s, 1)
        point = s(i, :);
        
        % Calculate Euclidean distance between the sensor and the point
        distance = norm(point - x);
        
        % Check if the point is within the sensing radius
        if distance <= Delta
            pointsWithinRadius = [pointsWithinRadius; point];
        end
    end
end