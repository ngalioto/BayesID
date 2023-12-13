addpath('../../../plotting');
addpath('../../../utils');

%% Plotting settings
fontSize = 18; lineWidth = 1.5;
plotSettings(fontSize, lineWidth)
sample_color = [0.3010 0.7450 0.9330 0.1];
mean_color = [0 0.4470 0.7410];
ms_color = [0.8500 0.3250 0.0980];
true_color = [0 0 0];
data_color = [0 0 0];

close all;
% load('WH_samples.mat')
load('whResults.mat');

%% Get output estimates from samples
fvec = @(x,u,theta)simple_res_net([x;u.*ones(p,size(x,2))],theta(indF),n+p,n,num_nodes);
hvec = @(x,u)simple_res_net([x;u.*ones(p,size(x,2))],theta_map(indH),n+p,m,num_nodes);

T_sim = T+1;
burn_in = 2e4;
num_samples = size(samples,2);
num_plottedSamples = 100; % total number of samples to use
sample_step = (num_samples - burn_in) / num_plottedSamples;
ysamples = zeros((num_samples - burn_in) / sample_step, T_sim);
parfor ii = 1:(num_samples - burn_in) / sample_step
    sample_ii = samples(:,burn_in+ii*sample_step);
    theta = [sample_ii([indx0,indF]); theta_map(indH); sample_ii((indH(end)+1):end)];
    ysamples(ii,:) = simulate(x0(theta).val,@(x,u)fvec(x,u,theta),hvec,u_test,T_sim);
    ii
end
% Point estimate
ymean = mean(ysamples*yscale+yshift);

%% Plot testing period
ii_plot = 1.778e5+1:1:1.788e5;
smp = plot(ii_plot,ysamples(2:2:end,ii_plot)'*yscale+yshift, 'Color', sample_color);
hold on
avg = plot(ii_plot,mean(ysamples(:,ii_plot)'*yscale+yshift,2), 'Color', mean_color);
ms = plot(ii_plot,sota_test(ii_plot-1e5),'Color',ms_color);
true = plot(ii_plot,yfull(ii_plot),'--','Color',true_color);
xlabel('Time Index')
ylabel('$y (V)$')
legend([smp(1),avg,ms,true], 'Sample', 'Mean', 'MS', 'Original')

%% Plot training period
figure
ii_plot = 1:1e3;
data = plot(ii_plot, y_train*yscale+yshift, '.', 'Color', data_color, 'MarkerSize', 10);
hold on;
smp = plot(ii_plot,ysamples(2:2:end,ii_plot)'*yscale+yshift, 'Color', sample_color);
hold on
avg = plot(ii_plot,mean(ysamples(:,ii_plot)'*yscale+yshift,2), 'Color', mean_color);
ms = plot(ii_plot,sota_train,'Color',ms_color);
true = plot(ii_plot,yfull(ii_plot),'--','Color',true_color);
xlabel('Time Index')
ylabel('$y (V)$')
legend([smp(1),avg,ms,true,data], 'Sample', 'Mean', 'MS', 'Original', 'Data')

%% Plot error in time domain
figure
ii_test = 1e5+1:1:1.788e5;
ii_plot = ii_test;
org = plot(ii_plot,yfull(ii_test),'k');
hold on
ms = plot(ii_plot,yfull(ii_test)-sota_test);
avg = plot(ii_plot,yfull(ii_test)-ymean(ii_test),'Color',mean_color);
xlabel('Time Index')
ylabel('$y (V)$')
xticks(0:1e4:8e4+1e5);
legend([org,ms,avg], 'Original','MS Error','Mean Error',...
    'Location','Best','Orientation','Horizontal')

%% Plot error in frequency domain
figure
nskip = 151; %skip transient
[freq,dB] = getPSD(yfull(ii_test(nskip:end)));
plot(freq,dB,'k.','MarkerSize',2);
hold on
[freq,dB] = getPSD(yfull(ii_test(nskip:end))-sota_test(nskip:end));
plot(freq,dB,'k.','MarkerSize',2,'Color',ms_color);
[freq,dB] = getPSD(yfull(ii_test(nskip:end))-ymean(ii_test(nskip:end)));
plot(freq,dB,'k.','MarkerSize',2,'Color',mean_color);
ylim([-160 -30]);
yticks(-160:20:-30);
xlim([0 freq(end)]);
xticks(0:5:35);
xlabel('kHz');
ylabel('dB');
org = plot(-1,1,'k.','MarkerSize',15);
ms = plot(-1,1,'k.','MarkerSize',15,'Color',ms_color);
avg = plot(-1,1,'k.','MarkerSize',15,'Color',mean_color);
legend([org,ms,avg],'Original', 'MS', 'Mean');

function [freq,dB] = getPSD(y)
    ydft = fft(y);
    ypsd = abs(ydft) / length(ydft);
    ypsd = ypsd(1:length(ypsd)/2+1);
    freq = (0:length(ypsd)-1)/1e3;
    dB = 20*log10(ypsd);
end