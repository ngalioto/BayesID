function [m,P] = updateKF(m,P,C,D,R,y,u)
    v = y - C*m;
    if (~isempty(D))
        v = v - D*u;
    end
    
    S = C*P*C' + R;
    Sinv = inv(S);
    K = P*C'*Sinv;

    m = m + K*v;
    P = P - K*C*P;
end