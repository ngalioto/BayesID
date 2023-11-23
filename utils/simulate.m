function [yhat,xhat] = simulate(x0,f,h,u,T)
    xhat = zeros(size(x0,1),T);
    xhat(:,1) = x0(:,1);
    for ii = 2:T
        xhat(:,ii) = f(xhat(:,ii-1),u(:,ii-1));
    end
    yhat = h(xhat, u);
end