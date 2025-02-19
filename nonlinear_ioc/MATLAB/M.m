function ret = M(states,phi_x)
    T = size(states,1);
    ret = [];
    for i = 2:T
        ret = [ret;phi_x(states(i,1),states(i,2))'];
    end
end