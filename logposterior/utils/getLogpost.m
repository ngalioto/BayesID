function [lp,PH,Sinv,Sinvv] = getLogpost(lp,m,P,H,R,y,Idy,D,u)
    v = y - H*m;
    if (nargin == 9)
        v = v - D*u;
    end
    
    PH = P*H';
    S = H*PH + R;
    Sinv = Idy / S;
    Sinvv = Sinv*v;

    lp = lp - 0.5*log(det(S)) - 0.5*(v'*Sinvv);
end