function [X,m,P] = updateEnKF(X,h,sqrtR,y,u)
    n = size(X,2);
    if (nargin(h) == 2)
        Y = h(X,u);
    else
        Y = h(X);
    end
    Y = Y + sqrtR*randn(size(Y));
    mu = mean(Y,2);
    Y0 = Y - mu;
    
    m = mean(X,2);
    U = sum((X-m) * Y0', 2)/(n-1);
    S = sum(Y0*Y0', 2) / (n-1);
    K = U/S;
    
    X = X + K*(y - Y);

    if (nargout > 1)
        m = mean(X, 2);
        if (nargout > 2)
            P = cov(X');
        end
    end
end