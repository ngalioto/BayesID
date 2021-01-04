function [xout, Pout,mu,S,err] = ukfUpdate(xin, Pin, y, R, n, h, lambda,Wm,Wc,eps)
    m = size(y, 1);
    [X,err] = formSigmaPoints(xin, Pin, n, lambda);
    if (err == 0)
        Yhat = zeros(m,size(X,2));
        for i = 1:2*n+1
            Yhat(:,i) = h(X(:,i));
        end

        mu = sum(Wm .* Yhat,2);
        S = Wc .* (Yhat - mu)*(Yhat - mu)' + R;
        C = Wc .* (X - xin)*(Yhat - mu)';

        K = C/S;
        xout = xin + K * (y - mu);
        Pout = Pin - K*S*K' + eps*eye(n);
    else
        xout = xin;
        Pout = Pin;
        mu = 0;
        S = 0;
    end
end