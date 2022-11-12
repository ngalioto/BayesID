function [lp,U,Sinv,Sinvv] = getLogpost_UKF(lp,n,m,P,R,h,u,y,Idy,Wm,Wc,lambda)
    [sigmaPts,err] = formSigmaPoints(m, P, n, lambda);
    if (err == 0)
        resx = sigmaPts - m;
        if (isempty(u))
            sigmaPts = h(sigmaPts);
        else
            sigmaPts = h(sigmaPts,u);
        end
        mu = sum(sigmaPts .* Wm,2);
        resy = sigmaPts - mu;
        v = y - mu;
        
        S = Wc .* resy*resy' + R;
        U = Wc .* resx*resy';
        Sinv = Idy / S;
        Sinvv   = Sinv*v;
        
        lp = lp - 0.5*log(det(S)) - 0.5*(v'*Sinvv);
    else
        U = 0; Sinv = 0; Sinvv = 0;
    end
end