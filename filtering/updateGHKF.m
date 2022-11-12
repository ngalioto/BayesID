function [m, P,v,S,Sinv,err] = updateGHKF(m, P, h, R, y, u, xi, W)
    [X,err] = formGHSigmaPoints(m, P, xi);
    if (err == 0)
        if (nargin(h) == 1)
            Yhat = h(X);
        else
            Yhat = h(X,u);
        end

        mu = sum(W .* Yhat,2);
        deltaY = Yhat-mu;
        S = (W.*deltaY)*deltaY' + R;
        U = W .* (X - m)*deltaY';

        K = U/S;
        v = y - mu;
        m = m + K*v;
        P = P - K*U';
    else
        v = 0;
        S = 0;
        Sinv = 0;
    end
end