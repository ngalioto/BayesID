function logpost = ukflp(theta, m, C, f, h, Sigma, Gamma, y, prior,alpha,beta,kappa,eps)
    T = size(y,2);
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
            [m,C,v,S,Sinv,err] = ukfUpdate(m, C, y(:,i), Gamma, n, h, lambda, Wm, Wc, eps);
            if (err ~= 0)
                logpost = -Inf;
                break;
            end
            logpost = logpost - 0.5*log(det(S)) - 0.5*v'*Sinv*v;
        end
    end
end