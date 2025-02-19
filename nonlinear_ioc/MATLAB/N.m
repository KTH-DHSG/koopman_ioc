function ret = N(controls,phi_u)
    T = size(controls,1);
    ret = [];
    for i = 1:T-1
        ret = [ret;phi_u(controls(i,1),controls(i,2))'];
    end
end