function yhat = simStochasticABC(x0, P0, A, B, C, Q, R, u, y)
addpath('C:\Users\Nick\Documents\MATLAB\nickfiltering\BayesianID\filtering');

    T_train = size(y,2);
    T_test = size(u,2);
    n = length(x0);
    ii_cur = 1:n;
    xhat = zeros(n,T_test+1); xhat(:,1) = x0;
    P = zeros(n,(T_test+1)*n); P(:,ii_cur) = P0;
    q = mvnrnd(zeros(n,1),Q,T_test)';
    for i = 1:T_test
        if (i < T_train)
            [xhat(:,i),P(:,ii_cur)] = updateKF(xhat(:,i),P(:,ii_cur),C,[],R,y(:,i));
        end
        ii_prev = ii_cur;
        ii_cur = ii_prev + n;
        [xhat(:,i+1),P(:,ii_cur)] = predictKF(xhat(:,i),P(:,ii_prev), A, B, Q, u(:,i));
        xhat(:,i+1) = xhat(:,i+1) + q(:,i);
    end       
    yhat = C*xhat;
end