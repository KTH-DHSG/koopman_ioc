function ret = B(states,controls,f_u)
    T = size(states,1);
    n = size(states,2);
    m = size(controls,2);
    ret = zeros(m*T,n*T);
    for i = 1:T
        ret(m*(i-1)+1:m*i,n*(i-1)+1:n*i) = f_u()';
    end
end


