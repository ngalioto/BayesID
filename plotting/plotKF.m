function xplots = plotKF(x,P,A,B,C,D,Q,R,y,u,t,Color)
addpath('../filtering')
if (nargin < 12)
    Color = [0 0.4470 0.7410 0.05];
end
    n = length(x);
    [m,Tdata] = size(y);
    Tsim = length(t);
    setText(16,'latex');
    xplots = cell(1,n);
    x = [x,zeros(n,Tsim-1)];
    stddev = zeros(n,Tsim);

    [x(:,1),P] = updateKF(x(:,1),P,C,D,R,y(:,1),u(:,1));
    stddev(:,1) = sqrt(diag(P));
    for k = 2:length(t)
        if (isempty(u))
            [x(:,k),P] = predictKF(x(:,k-1),P,A,B,Q);
            if (k <= Tdata)
                [x(:,k),P] = updateKF(x(:,k),P,C,D,R,y(:,k));
            end
        else
            [x(:,k),P] = predict(x(:,k-1),P,A,B,Q,u(:,k-1));
            if (k <= Tdata)
                [x(:,k),P] = updateKF(x(:,k),P,C,D,R,y(:,k),u(:,k));
            end
        end
        stddev(:,k) = sqrt(diag(P));
    end
    t_fill = [t,fliplr(t)];
    stddev_fill = [x+2*stddev, fliplr(x-2*stddev)];
    for i = 1:m
        xplots{i} = figure; hold on;
        x_mean = plot(t,x(i,:),'Color',Color(1:3));
        post = fill(t_fill, stddev_fill(i,:), 'Color', Color);
        legend([x_mean, post], 'Mean', '$\pm2\sigma$');
    end
end