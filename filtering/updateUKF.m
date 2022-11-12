function [m,P,v,S,Sinv,err] = updateUKF(m, P, n, h, R, y, u, lambda,Wm,Wc,eps)
%     m = size(y, 1);
    [X,err] = formSigmaPoints(m, P, n, lambda);
    if (err == 0)
        if (nargin(h) == 1)
            Yhat = h(X);
        else
            Yhat = h(X,u);
        end

        mu = sum(Wm .* Yhat,2);
        deltaY = Yhat-mu;
        S = (Wc.*deltaY)*deltaY' + R;
        U = Wc .* (X - m)*deltaY';

%         Sinv = eye(m) / S;
        K = U/S;
        v = y - mu;
        m = m + K*v;
        P = P - K*U';% + eps*eye(n);
    else
        v = 0;
        S = 0;
        Sinv = 0;
    end
end