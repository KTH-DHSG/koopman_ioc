function ret = N(controls,phi_u)
    T = size(controls,1);
    ret = [];
    for i = 1:T
        ret = [ret;phi_u()'];
    end
end