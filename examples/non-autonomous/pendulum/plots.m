addpath('../../../plotting');
%% Plotting settings
fontSize = 18; lineWidth = 1.5;
plotSettings(fontSize, lineWidth);

sample_color = [0.3010 0.7450 0.9330 0.05];
mean_color = [0 0.4470 0.7410];
ls_color = [0.8500 0.3250 0.0980];
true_color = [0 0 0];
data_color = [0 0 0];

close all;
load('pendResults.mat');

%% Plotting contour plots
% First post-process results
MSElsq_post = removeWorst(MSElsq, 1);

BayesMSE = getMeanMSE(MSE,1);
predBayesMSE = getMeanMSE(MSE,2);
lsqMSE = getMeanMSE(MSElsq_post,1);
predlsqMSE = getMeanMSE(MSElsq_post,2);

caxisMAP = [-12, -3];
TicksMAP = -12:-3;
TickLabelsMAP = {'-12','','','-9','','','-6','','','-3'};
caxisLS = [-2, -1];
TicksLS = -2:0.2:-1;

figure;
contourf(noise_ratio, dt, log10(BayesMSE))
caxis(caxisMAP)
xlabel('Noise Ratio');
ylabel('$\Delta t$');

figure;
contourf(noise_ratio, dt, log10(predBayesMSE))
caxis(caxisMAP)
colorbar('Ticks',TicksMAP, 'TickLabels', TickLabelsMAP);
xlabel('Noise Ratio');
yticklabels("");

figure;
contourf(noise_ratio, dt, log10(lsqMSE))
caxis(caxisLS);
xlabel('Noise Ratio');
ylabel('$\Delta t$');

figure;
contourf(noise_ratio, dt, log10(predlsqMSE))
caxis(caxisLS)
colorbar('Ticks',TicksLS);
xlabel('Noise Ratio');
yticklabels("");

%% Plot trajectories using posterior samples
% Assume 'pendResults.mat' already loaded

% select options to determine plot
% data options are: 'noise' and 'timestep'
% response options are: 'impulse' and 'forced'
data_choice = 'noise';
response = 'forced';

switch data_choice
    case 'noise'
        load('samples_noise.mat');
        ii_dt = 1;
        ii_noise = 9;
        iter = 21;
    case 'timestep'
        load('samples_timestep.mat');
        ii_dt = 9;
        ii_noise = 1;
        iter = 84;
    otherwise
        error('Please define "data_choice" to be either "noise" or "timestep"')
end
% define constants
T_data = tf / dt(ii_dt);
T_sim = 2*T_data + 1; % train + test period
t = 0:dt(ii_dt):2*tf;
num_samples = size(samples,2);
Atrue = expm(Acon*dt(ii_dt));
% load results
map = THETA{ii_dt,ii_noise}(:,iter);
thlsq = LSQ{ii_dt,ii_noise}(:,iter);
y = Yfull{ii_dt,ii_noise}(iter,:);

switch response
    case 'impulse'
        u = [1/dt(ii_dt),zeros(1,T_sim-1)];
    case 'forced'
        u = Ufull{ii_dt, ii_noise}(iter,:);
    otherwise
        error('Please define "response" to be either "impulse" or "forced"');
end

figure; hold on;
burn_in = 1e5; num_plottedSamples = 100; 
sample_step = floor((num_samples-burn_in) / num_plottedSamples);
ysamples = zeros(m,2*T_data+1,100);
for i = 1:num_plottedSamples
    sample_idx = burn_in + sample_step*i;
    theta = [samples(1:6,sample_idx); map(7:10); samples(7:9,sample_idx)];
    switch response
        case 'impulse'
            ysamples(:,:,i) = simulate_LTI(x0true,A(theta).val,B(theta).val,C(theta).val,[],u,T_sim);
        case 'forced'
            ysamples(:,:,i) = simulate_LTI(x0(theta).val,A(theta).val,B(theta).val,C(theta).val,[],u,T_sim);
    end
end

% Get point estimates
ymap = simulate_LTI(x0(map).val,A(map).val,B(map).val,C(map).val,[],u,T_sim);
ymean = mean(ysamples,3);
ylsq = simulate_LTI(x0true,reshape(thlsq(1:4),[2,2]), thlsq(5:6), thlsq(7:8)', [], u,T_sim);
ytrue = simulate_LTI(x0true,Atrue,Btrue,Ctrue,[],u,T_sim);

% Plot outputs
plot(t,squeeze(ysamples), 'Color', sample_color); 
plot(t, ymean, 'Color', mean_color); 
plot(t,ylsq, 'Color', ls_color);
plot(t,ytrue, 'Color', true_color, 'LineStyle', ':');
xlabel('Time (s)');
switch response
    case 'impulse'
        ylabel('Impulse response');
        xlim([0 tf]);
    case 'forced'
        plot(t(1:T_data),y(:,1:T_data), '.', 'Color', data_color, 'MarkerSize',10);
        ylabel('$y$');
end

%% Helper functions
function MSE = removeWorst(MSE, num_remove)
    N = length(MSE{1,1});
    for ii = 1:size(MSE,1)
        for jj = 1:size(MSE,2)
            for kk = 1:num_remove
                [~,I] = max(MSE{ii,jj}(2,:));
                if I == 1
                    MSE{ii,jj} = MSE{ii,jj}(:,2:end);
                elseif I == N
                    MSE{ii,jj} = MSE{ii,jj}(:,1:end-1);
                else
                    MSE{ii,jj} = MSE{ii,jj}(:,[1:I-1, I+1:end]);
                end
            end
        end
    end
end

function meanMSE = getMeanMSE(MSE,dim)
    [n,m] = size(MSE);
    meanMSE = zeros(n,m);
    for ii = 1:n
        for jj = 1:m
            meanMSE(ii,jj) = mean(MSE{ii,jj}(dim,:));
        end
    end
end