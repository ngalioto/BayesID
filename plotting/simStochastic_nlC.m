function yhat = simStochastic_nlC(x0, P0, f, C, Q, R, u, y,lambda,Wm,Wc)
    T_train = size(y,2);
    T_test = size(u,2);
    n = length(x0);
    ii_cur = 1:n;
    xhat = zeros(n,T_test+1); xhat(:,1) = x0;
    P = zeros(n,(T_test+1)*n); P(:,ii_cur) = P0;
    for i = 1:T_test
        if (i < T_train)
            [xhat(:,i),P(:,ii_cur)] = updateKF(xhat(:,i),P(:,ii_cur),C,[],R,y(:,i),[]);
            ii_prev = ii_cur;
            ii_cur = ii_prev + n;
            [xhat(:,i+1),P(:,ii_cur)] = predictUKF(n, xhat(:,i),P(:,ii_prev), @(x)f(x,u(:,i)), Q, u, Wm, Wc, lambda);
        else
            xhat(:,i+1) = f(xhat(:,i),u(:,i));
        end
        xhat(:,i+1) = xhat(:,i+1) + mvnrnd(zeros(n,1),Q)';
    end       
    yhat = C*xhat;
end