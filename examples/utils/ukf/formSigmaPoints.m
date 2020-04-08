function [xout,err] = formSigmaPoints(xin, Pin, n, lambda)
	[~,err] = chol(Pin);
    if (err == 0)
        sqrtP = sqrt(n+lambda)*chol(Pin);
        xout = zeros(length(xin), 2*n+1);
        xout(:, 1) = xin;
        for i = 1:n
            xout(:,i+1) = xin + sqrtP(i,:)';
            xout(:,i+n+1) = xin - sqrtP(i,:)';
        end
    else
        xout = xin;
    end
end