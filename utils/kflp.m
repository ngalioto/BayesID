function logpost = kflp(theta, m, C, A, H, Sigma, Gamma, y, prior,I)
    T = size(y,2);
    
    logpost = prior(theta);
    for i = 1:T
        m = A*m;
        C = A*C*A' + Sigma;
        
        HC = H*C;
        mu = H*m;
        v = y(:,i) - mu;
        S = HC*H' + Gamma;
        Sinv = I/S;
        
        logpost = logpost - 0.5*log(det(S)) - 0.5*v'*Sinv*v;
        
        K = C*H' * Sinv;
        m = m + K*v;
        C = C - K*HC;
    end
end