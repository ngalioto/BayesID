function xplots = plotNLDynUKF(x,P,f,C,D,Q,R,y,u,t,alpha,beta,kappa,Color)
addpath('C:/Users/Nick/Documents/MATLAB/nickfiltering/BayesianID/filtering')
if (nargin < 14)
    Color = [0 0.4470 0.7410 0.05];
end
    n = length(x);
    [m,Tdata] = size(y);
    [Wm, Wc, lambda] = formWeights(n, alpha, beta, kappa);

    Tsim = length(t);
    setText(16,'latex');
    set(0,'DefaultLineLineWidth',1.5)
    xplots = cell(1,n);
    x = [x,zeros(n,Tsim-1)];
    stddev = zeros(n,Tsim);

%     [x(:,1),P] = update(x(:,1),P,C,D,R,y(:,1),u(:,1));
    stddev(:,1) = sqrt(diag(P));
    for k = 2:length(t)
        if (isempty(u))
            [x(:,k),P] = predictUKF(n,x(:,k-1),P,f,Q,[],Wm,Wc,lambda);
            if (k <= Tdata)
                [x(:,k),P] = updateKF(x(:,k),P,C,D,R,y(:,k));
            end
        else
            [x(:,k),P] = predictUKF(n,x(:,k-1),P,f,Q,u(:,k-1),Wm,Wc,lambda);
            if (k <= Tdata)
                [x(:,k),P] = updateKF(x(:,k),P,C,D,R,y(:,k),u(:,k));
            end
        end
        stddev(:,k) = sqrt(diag(P));
    end
    t_fill = [t,fliplr(t)];
    stddev_fill = [x+2*stddev, fliplr(x-2*stddev)];
    for i = 1:n
        xplots{i} = figure; hold on;
        x_mean = plot(t,x(i,:),'Color',Color(1:3));
        post = fill(t_fill, stddev_fill(i,:), Color(1:3),'FaceAlpha',Color(4));
        legend([x_mean, post], 'Mean', '$\pm2\sigma$');
    end
end