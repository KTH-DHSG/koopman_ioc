function ret = V(states,controls,f_x)
    T = size(states,1);
    n = size(states,2);
    ret = zeros(n*(T-1),n);
    ret(end-n+1:end,:) = f_x(states(T,3),controls(T,1))';
end