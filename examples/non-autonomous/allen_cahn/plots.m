addpath('../../../plotting');
addpath('../../../utils');

%% Plotting settings
fontSize = 18; lineWidth = 1.5;
plotSettings(fontSize, lineWidth)
sample_color = [0.3010 0.7450 0.9330 0.05];
mean_color = [0 0.4470 0.7410];
ls_color = [0.8500 0.3250 0.0980];
true_color = [0 0 0];
data_color = [0 0 0];

close all;
load('acResults.mat');

%% Get output estimates from samples
burn_in = 5e4;
num_samples = size(samples,2);
num_plottedSamples = 100; % total number of samples to use
sample_step = (num_samples - burn_in) / num_plottedSamples; %subsampling interval
T_sim = 2*T + 1;
t = 0:dt:dt*(T_sim-1);
yDeterministic = zeros(num_plottedSamples, T_sim);
yStochastic = zeros(num_plottedSamples, T_sim);
for ii = 1:num_plottedSamples
    sample_idx = burn_in + sample_step*ii;
    theta = samples(:, sample_idx);
    yDeterministic(ii,:) = yscale*simulate(x0(theta).val, @(x,u)fvec(x,u,theta),...
        @(x,u)H(theta).val*x, u, T_sim)+yshift;
    yStochastic(ii,:) = yscale*simStochastic_nlC(x0(theta).val,P0(theta).val,...
            @(x,u)fvec(x,u,theta),H(theta).val,Q(theta).val,R(theta).val,...
            u(1:(T_sim-1)),y,lambda,Wm,Wc)+yshift;
end
% Point estimate
yls = yscale*simulate(x0(theta_ls).val, @(x,u)fvec(x,u,theta_ls),...
    @(x,u)H(theta_ls).val*x, u, T_sim)+yshift;

%% Deterministic simulation
figure; hold on;
plot(t,yDeterministic','Color',sample_color);
avg = plot(t,mean(yDeterministic),'Color',mean_color);
ls = plot(t,yls,'Color',ls_color);
truth = plot(t,y_test,':','Color',true_color);
data = plot(t(1:T+1),yraw,'.','MarkerSize',10,'Color',data_color);
legend([avg,ls,truth,data],'Mean','LS','Truth','Data')
xlabel('Time (s)')
ylabel('$y$')
set(gcf,'Position',[137 250.3333 830 288]);

%% Stochastic simulation
figure; hold on;
plot(t,yStochastic','Color',sample_color);
avg = plot(t,mean(yStochastic),'Color',mean_color);
truth = plot(t,y_test,':','Color',true_color);
data = plot(t(1:T+1),yraw,'.','MarkerSize',10,'Color',data_color);
legend([avg,truth,data],'Mean','Truth','Data')
xlabel('Time (s)')
ylabel('$y$')
set(gcf,'Position',[137 250.3333 830 288]);