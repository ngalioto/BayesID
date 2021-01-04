function [xout, Pout,err] = ukfPredict(xin, Pin, Q, n, f, lambda, Wm, Wc)
    [X,err] = formSigmaPoints(xin, Pin, n, lambda);
    if (err == 0)
    Xhat = zeros(size(X));
    for i = 1:2*n+1
        Xhat(:,i) = f(X(:,i));
    end
    
    xout = sum(Xhat .* Wm,2);
    Pout = Wc .* (Xhat - xout)*(Xhat - xout)' + Q;
    else
        xout = xin;
        Pout = Pin;
    end
end
