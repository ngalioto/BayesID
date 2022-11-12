function [m,P] = updateEKF(m,P,h,C,R,y,u)
    if (nargin(h) == 2)
        mu = h(m,u);
    else
        mu = h(m);
    end
    v = y - mu;
    
    S = C*P*C' + R;
    U = P*C';
    K = U/S;

    m = m + K*v;
    P = P - K*U';
end