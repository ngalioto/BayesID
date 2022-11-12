function [X,m,P] = predictEnKF(X,f,sqrtQ,u)
    if (nargin(f) == 2)
        X = f(X,u);
    else
        X = f(X);
    end
    X = X + sqrtQ*randn(size(X));
    
    if (nargout > 1)
        m = mean(X, 2);
        if (nargout > 2)
            P = cov(X');
        end
    end
end