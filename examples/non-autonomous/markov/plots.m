addpath('../../../../plotting')
clear; close all;
load('smallDim.mat'); %choose dataset here

fontSize = 30; lineWidth = 1.5;
plotSettings(fontSize, lineWidth);

colorQR = [0 0.4470 0.7410];
colorLS = [0.8500 0.3250 0.0980];
colorMAP = [0.9290 0.6940 0.1250];
xfill = [Karr, fliplr(Karr)];
xfill_map = [Karr_map, fliplr(Karr_map)];
kNoise = size(LSerr, 3);
for k = 1:kNoise
    figure; hold on;
    loc = mean(LSerr(:,:,k),1);
    std = sqrt(var(LSerr(:,:,k),0,1));
    ls = plot(Karr,loc,'-', 'Color', colorLS);
    fill(xfill, [loc-std, fliplr(loc+std)], colorLS, 'FaceAlpha', 0.05);

    loc = mean(GLSerr(:,:,k),1);
    std = sqrt(var(GLSerr(:,:,k),0,1));
    gls = plot(Karr,loc,'-', 'Color', colorMAP);
    fill(xfill, [loc-std, fliplr(loc+std)], colorMAP, 'FaceAlpha', 0.05);
    
    loc = mean(MAPerr(:,:,k),1);
    std = sqrt(var(MAPerr(:,:,k),0,1));
    map = plot(Karr_map,loc,'.-', 'Color', colorQR,'MarkerSize',10);
    fill(xfill_map, [loc-std, fliplr(loc+std)], colorQR, 'FaceAlpha', 0.05);
    grid on
    ylim([0 40])
    xlabel('Number of Data')
    if k == 1
        ylabel('$|| \hat{\mathbf{G}}-\mathbf{G} ||_2$','Interpreter','Latex')
        legend([map,ls,gls],'MAP','LS','GLS')
    else
        yticklabels("");
    end
    xlim([0 K]);
    box on;
end