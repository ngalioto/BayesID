function yplots = plotPosterior(samples,x0,P0,A,B,C,Q,R,y,u,t,Color)
if (nargin < 12)
%     Color = [0 0.4470 0.7410 0.05];
    Color = [0.3010 0.7450 0.9330 0.05];
end
    [~,Tdata] = size(y);
    m = size(C(samples(:,1)).mat,1);
    N = size(samples,2);
    setText(16,'latex');
    yplots = cell(1,m);
    y_mean = zeros(m,size(u,2)+1);
    for i = 1:m
        yplots{i} = figure;
        hold on;
    end
    ii = 1;
    for i = 1:N
        for k = 1:1
            yhat = simStochasticABC(x0(samples(:,i)).vec(:,1),P0(samples(:,i)).mat,...
                A(samples(:,i)).mat,B(samples(:,i)).mat,C(samples(:,i)).mat,...
                Q(samples(:,i)).mat,R(samples(:,i)).mat,u,y);
            y_mean = (yhat + (ii-1)*y_mean) ./ ii;
            ii = ii + 1;
            for j = 1:m
                figure(yplots{j});
                post = plot(t,yhat(j,:),'Color',Color);
            end
        end
    end
    for i = 1:m
        figure(yplots{i});
        if (~isempty(y))
            data = plot(t(1:Tdata),y(i,:),'k.','MarkerSize',10);
        end
        avg = plot(t, y_mean,'Color', [0 0.4470 0.7410]);
        if (~isempty(y))
            legend([post, avg, data], 'Sample', 'Mean', 'Data');
        else
            legend([post, avg], 'Sample', 'Mean');
        end
    end
end