function propC = condHess(hessian)
    n = size(hessian,1);
    for i = 1:n^2
        if (isnan(hessian(i)))
            hessian(i) = 1e10;
        elseif hessian(i) == 0
            hessian(i) = 1e-8;
        end
    end
    i = -16;
    while (sum(eig(inv(hessian) + 10^i*eye(n)) <= 0) > 0)
        i = i + 1;
    end
    propC = inv(hessian) + 10^i*eye(n);
end