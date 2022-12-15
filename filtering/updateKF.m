function [m,P] = updateKF(m,P,C,D,R,y,u)
    v = y - C*m;
    if (~isempty(D))
        v = v - D*u;
    end
    

    PC = P*C';
    S = C*PC + R;
    K = PC/S;

    m = m + K*v;
    P = P - K*PC';
end