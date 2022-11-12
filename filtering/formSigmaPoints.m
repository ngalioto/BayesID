function [xout,err] = formSigmaPoints(xin, P, n, lambda)
	[L,err] = chol(P,'lower');
    if (err == 0)
        scaledL = sqrt(n+lambda)*L;
        xout = zeros(n,2*n+1);
        xout(:,1) = xin;
        xout(:,2:n+1) = xin + scaledL;
        xout(:,n+2:end) = xin - scaledL;
    else
        xout = xin;
    end
end