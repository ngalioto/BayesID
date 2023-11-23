addpath('../../../plotting');
%% Plotting settings
sampleColor = [0.3010 0.7450 0.9330 0.05];
mapColor = [0 0.4470 0.7410];
lsqColor = [0.4940 0.1840 0.5560];
msColor = [0.8500 0.3250 0.0980];
truthColor = 'k';

fontSize = 16; lineWidth = 1;
plotSettings(fontSize, lineWidth);

close all;
load('duffingResults.mat');

%% Get output estimates from samples
T_sim = 2*T;
burn_in = 5e5;
num_samples = size(samples,2);
num_plottedSamples = 50; % total number of samples to use
sample_step = (num_samples - burn_in) / num_plottedSamples; %subsampling interval
ysamples = zeros(num_plottedSamples, T_sim);
for i = 1:num_plottedSamples
    theta = samples(:,burn_in+i*sample_step);
    ysamples(i,:) = simulate(x0(theta).val,@(x,u)fvec(x,u,theta),...
        @(x,u)H(theta).val*x,u,T_sim);
end
% Point estimates
ymap = simulate(x0(theta_map).val,@(x,u)fvec(x,u,theta_map),@(x,u)H(theta_map).val*x,u,T_sim);
yls = simulate(x0(theta_ls).val,@(x,u)fvec(x,u,theta_ls),@(x,u)H(theta_ls).val*x,u,T_sim);
yms = simulate(theta_ms(pdyn+1:pdyn+n),@(x,u)fvec(x,u,theta_ms(1:pdyn)),...
    @(x,u)H(theta_ms(1:pdyn)).val*x,u,T_sim);

%% Stochastic simulation
tdata = t(1:T);
plotPosterior_nl(samples(:,burn_in+sample_step:sample_step:end),...
    x0,P0,fvec,H,Q,R,y,u(:,1:T_sim),t,theta_map,lambda,Wm,Wc);
data = plot(tdata,y,'k.');
truth = plot(t(1:2:T),x(1,1:2:T), 'Color', truthColor,'LineStyle',':');
legend([data, truth], 'Data', 'Truth')
xlabel('Time (s)');
ylabel('$y$')

%% Deterministic simulation Bayes
figure; hold on;
smp = plot(t(1:2:T),ysamples(:,1:2:T)', 'Color', sampleColor);
map_plot = plot(t(1:2:T),ymap(1:2:T), 'Color', mapColor);
truth = plot(t(1:2:T),x(1,1:2:T), 'Color', truthColor,'LineStyle',':');
legend([smp(1), map_plot, truth], 'Sample', 'MAP', 'Truth')
xlabel('Time (s)')
ylabel('$y$')

%% Deterministic simulation LS/MS
figure; hold on;
ls_plot = plot(t(1:2:T),yls(1:2:T), 'Color', lsqColor);
ms_plot = plot(t(1:2:T),yms(1:2:T), 'Color', msColor);
truth = plot(t(1:2:T),x(1,1:2:T), 'Color', truthColor,'LineStyle',':');
legend([ls_plot, ms_plot, truth], 'LS', 'MS', 'Truth')
xlabel('Time (s)')
ylabel('$y$')

%% Phase plots
% MAP phase
figure
plot(ymap(1:end-1),ymap(2:end),'Color',mapColor);
xlabel('$y(t)$');
ylabel('$y(t+\Delta t)$');

% Truth phase
figure
plot(x(1,1:end-1),x(1,2:end),'Color',truthColor);
xlabel('$y(t)$');
ylabel('$y(t+\Delta t)$');

% MS phase
figure
plot(yms(1:end-1),yms(2:end),'Color',msColor);
xlabel('$y(t)$');
ylabel('$y(t+\Delta t)$');
xlim([-2 2])
ylim([-2 2])