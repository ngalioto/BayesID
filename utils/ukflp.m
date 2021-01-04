function logpost = ukflp(theta, m, C, f, h, Sigma, Gamma, y, prior,alpha,beta,kappa,eps)
    [d,T] = size(y);
    n = size(m,1);
    logpost = prior(theta);
    if (isfinite(logpost))
    
        lambda = alpha^2 * (n+kappa) - n;
        [Wm, Wc] = formWeights(n, lambda, alpha, beta);

        for i = 1:T
            [m, C,err] = ukfPredict(m, C, Sigma, n, f, lambda, Wm, Wc);
            if (err ~= 0)
                logpost = -Inf;
                break;
            end
            [m,C,mu,S,err] = ukfUpdate(m, C, y(:,i), Gamma, n, h, lambda, Wm, Wc, eps);
            if (err ~= 0)
                logpost = -Inf;
                break;
            end
            logpost = logpost - 0.5*log(det(S)) - 0.5*d*log(2*pi) -  ...
                0.5*(y(:,i)-mu)'/S*(y(:,i)-mu);
        end
    end
end