function [yplots,lines] = plotPosterior_nl(samples,x0,P0,f,C,Q,R,y,u,t,map,lambda,Wm,Wc,scale,shift,Color)
if (nargin < 17)
%     Color = [0 0.4470 0.7410 0.05];
    Color = [0.3010 0.7450 0.9330 0.05];
end
    [m,Tdata] = size(y);
    N = size(samples,2);
    plotSettings(16,1,'latex');
    yplots = cell(1,m);
    y_mean = zeros(m,size(u,2)+1); %%
    phase = cell(1,m);
    phaseL = cell(1,m);
    post = cell(1,m);
    data = cell(1,m);
    for i = 1:m
        yplots{i} = figure;
        hold on;
        phase{i} = figure; hold on;
    end
    y_mean = simStochastic_nlC(x0(map).vec(:,1),P0(map).mat,...
            @(x,u)f(x,u,map),C(map).mat,...
            Q(abs(map)).mat,R(map).mat,u,y,lambda,Wm,Wc);
    ii = 1;
    for i = 1:N
        yhat = simStochastic_nlC(x0(samples(:,i)).vec(:,1),P0(samples(:,i)).mat,...
            @(x,u)f(x,u,samples(:,i)),C(samples(:,i)).mat,...
            Q(abs(samples(:,i))).mat,R(samples(:,i)).mat,u,y,lambda,Wm,Wc);
%         yhat = scale*yhat + shift;
%         y_mean = (yhat + (ii-1)*y_mean) ./ ii; %%
        ii = ii+1;
        for j = 1:m
            figure(yplots{j});
            if (i == N)
                post{j} = plot(t,yhat(j,:),'Color',Color,'LineWidth',1);
                figure(phase{j});
                phaseL{j} = plot(yhat(j,1:end-1),yhat(j,2:end),'Color',Color,'LineWidth',1);
            else
                plot(t,yhat(j,:),'Color',Color,'LineWidth',1);
                figure(phase{j});
                plot(yhat(j,1:end-1),yhat(j,2:end),'Color',Color,'LineWidth',1);
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
            legend([post{i}, avg, data], 'Sample', 'MAP', 'Data');
        else
            legend([post{i}, avg], 'Sample', 'Mean');
        end
    end
    lines = {post,data,phaseL};
end