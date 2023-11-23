function [y,x] = simulate_LTI(x0,A,B,C,D,u,T)
    x = zeros(size(x0,1),T);
    x(:,1) = x0;
    useB = exist('B','var') && ~isempty(B);
    for i = 1:T-1
        x(:,i+1) = A*x(:,i);
        if useB
            x(:,i+1) = x(:,i+1) + B*u(:,i);
        end
    end
    y = C*x;
    if exist('D', 'var') && ~isempty(D)
        y = y + D*u;
    end
end