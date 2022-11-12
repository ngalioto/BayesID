% Adaptive covariance algorithm.  See Haario 2006.
function [xbar_n, C_n] = AdaptiveCov(samples, xbar_nn, C_nn, d, n, n0, sd, eps)
    if n > n0
        x_n = samples(:,n);
        xbar_n = ((n-1)*xbar_nn + x_n) / n;
        term1 = ((n-1)/n) * C_nn;
        term2 = n*(xbar_nn*xbar_nn') - (n+1)*(xbar_n*xbar_n');
        term3 = (x_n*x_n') + eps*eye(d);

        C_n = term1 + (sd/n)*(term2 + term3); 
    elseif n == n0
        xbar_n = mean(samples(:,1:n),2);
        C_n = cov(samples(:,1:n)');
    else
        C_n = C_nn;
        xbar_n = xbar_nn;
    end
end