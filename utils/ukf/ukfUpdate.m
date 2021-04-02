function [xout,Pout,v,S,Sinv,err] = ukfUpdate(xin, Pin, y, R, n, h, lambda,Wm,Wc,eps)
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
        
        Sinv = eye(m) / S;
        v = y - mu;
        K = C*Sinv;
        xout = xin + K*v;
        Pout = Pin - K*S*K' + eps*eye(n);
    else
        xout = xin;
        Pout = Pin;
        v = 0;
        S = 0;
        Sinv = 0;
    end
end