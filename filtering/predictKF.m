function [m,P] = predictKF(m,P,A,Q,B,u)
    m = A*m;
    P = A*P*A' + Q;
    if (nargin > 4)
        m = m + B*u;
    end
end