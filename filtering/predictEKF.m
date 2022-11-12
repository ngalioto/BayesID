function [m,P] = predictEKF(m,P,f,A,Q,u)
    if (nargin(f) == 2)
        m = f(m,u);
    else
        m = f(m);
    end
    P = A*P*A' + Q;
end