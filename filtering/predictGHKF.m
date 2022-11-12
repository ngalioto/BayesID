function [m, P, err] = predictGHKF(m, P, f, Q, u, xi, W)
    [X,err] = formGHSigmaPoints(m, P, xi);
    if (err == 0)
        if (nargin(f) == 1)
            X = f(X);
        else
            X = f(X,u);
        end

        m = sum(X .* W,2);
        delta = X - m;
        P = (W.*delta)*delta' + Q;
    end
end