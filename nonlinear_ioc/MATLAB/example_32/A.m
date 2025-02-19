function ret = A(states,controls,f_x)
    T = size(states,1);
    n = size(states,2);
    ret = eye(n*T);
    for i = 1:T-1
        ret(n*(i-1)+1:n*i,n*(i-1)+1:n*i) = eye(n);
        ret(n*(i-1)+1:n*i,n*i+1:n*(i+1)) = -f_x(states(i,1))';
    end
end


