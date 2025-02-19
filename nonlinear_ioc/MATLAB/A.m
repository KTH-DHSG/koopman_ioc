function ret = A(states,controls,f_x)
    T = size(states,1);
    n = size(states,2);
    ret = eye(n*(T-1));
    for i = 1:T-2
        ret(n*(i-1)+1:n*i,n*i+1:n*(i+1)) = -f_x(states(i+1,3),controls(i+1,1))';
    end
end


