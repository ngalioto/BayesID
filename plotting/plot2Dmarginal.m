function plt = plot2Dmarginal(samples)
    C = [0 0.4470 0.7410; 0.8500 0.3250 0.0980];

    % plt = figure;
    plt = 1;
    [N,~,M] = size(samples);
    for i = 1:N
        for j = 1:i
            subplot(N,N,(i-1)*N+j); hold on;
            if (j == i)
                for k = 1:M
                    histogram(samples(i,:,k),'FaceColor',C(1,:));
                end
            else
                for k = 1:M
    %                 scatter(samples(j,:,k), samples(i,:,k),5,'filled','MarkerFaceAlpha',alph);
                    hold on;ksdensity([samples(j,:,k)', samples(i,:,k)'],'PlotFcn','contour');
                    colormap('default');
                end
            end
            if (j >1)
                set(gca,'ytick',[]);
            end
            if (i~= N)
                set(gca,'xtick',[]);
            end
        end
    end
end