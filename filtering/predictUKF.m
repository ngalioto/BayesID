function [m, P, err] = predictUKF(n, m, P, f, Q, u, Wm, Wc, lambda)
    [X,err] = formSigmaPoints(m, P, n, lambda);
    if (err == 0)
        if (nargin(f) == 1)
            Xhat = f(X);
        else
            Xhat = f(X,u);
        end

        m = sum(Xhat .* Wm,2);
        delta = Xhat - m;
        P = (Wc.*delta)*delta' + Q;
    end
end