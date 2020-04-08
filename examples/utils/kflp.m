function logpost = kflp(theta, m, C, A, H, Sigma, Gamma, y, prior)
    [d,T] = size(y);
    
    logpost = prior(theta);
    for i = 1:T
        m = A*m;
        C = A*C*A' + Sigma;
        
        mu = H*m;
        S = H*C*H' + Gamma;
        
        logpost = logpost - 0.5*log(det(S)) - 0.5*d*log(2*pi) -  ...
            0.5*(y(:,i)-mu)'/S*(y(:,i)-mu);
        
        K = C*H' / S;
        m = m + K *(y(:,i)-H*m);
        C = C - K*H*C;
    end
end