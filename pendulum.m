clear all; close all;
addpath('utils');
rng(1);

% Time parameters
tf = 4;
T = 40;
dt = tf / T;
t = 0:dt:tf;
dt_fine = 0.01; % for data generation/simulating truth

% Dynamics model
n = 2;
x0 = [0.1; -0.5];
P0 = zeros(n);
pdyn = n^2; %number of dynamics parameters
pvar = 2; %number of variance parameters
F = @(theta)reshape(theta(1:pdyn), [n,n])';
Q = @(theta)theta(end-1)*eye(n);

% Pendulum system and initial conditions
g = 9.81;
L = 1;
A = [0 1; -g/L 0];
ptrue = reshape(expm(A*dt)', [pdyn, 1]);
pend = @(x)A*x;
% pend = @(x)[x(2); -g/L*sin(x(1))]; % nonlinear pendulum

% Measurement model
sigmaR = 0.1;
H = @(theta)eye(n);
m = size(H(ptrue),1);
R = @(theta)theta(end)*eye(m);
y = generateData(@(x)pend(x), x0, t, H(ptrue), sigmaR);

% Sampling parameters
num_samp = 1e3;

%%
% Log of the prior distribution
lambda = 0.1; %sparsity knob
logprior = @(theta)log(rhnpdf(theta(pdyn+1:end),zeros(pvar,1),eye(pvar))) - ...
    lambda*norm(theta(1:end-pvar),1);

% Optimization
objective = @(theta)-kflp(theta, x0, P0, F(theta), H(theta), ...
    Q(theta), R(theta), y, logprior);
theta_init = [zeros(pdyn,1); 0.1*ones(pvar,1)]; %anywhere where objective is defined
options = optimoptions('fmincon', 'MaxIterations', 100);
[theta0,~,~,~,~,~,~] = fmincon(objective, theta_init, [],[],[],[],...
    [-Inf*ones(pdyn,1);zeros(pvar,1)],[], [],options);

% Compute the Hessian
[theta0, ~,~,~,~, hessian] = fminunc(objective, theta0);

% Condition Hessian to ensure positive definiteness
propC = condHess(hessian);

%% Sample from posterior
[samples, acc] = BayesLin(y, theta0, propC, num_samp, x0, P0,...
    F, H, Q, R, logprior);

%%
[t_fine, x] = ode45(@(t,x)pend(x), 0:dt_fine:2*tf, x0);
x = x';

A = @(theta)logm(F(theta)) / dt;
N = 100;
ind = num_samp / N;
xpost = zeros(n,length(t_fine),N);
for i = 1:N
    [~, xhat] = ode45(@(t,x)A(samples(:,i*ind))*x, t_fine, x0);
    xpost(:,:,i) = xhat';
end

%%
close all;
plotResults(t_fine, x0, x, t(2:end), y, xpost);


function plotResults(t,x0, x, tdata, y, xpost)
    plottingPreferences();

    n = length(x0);

    % Get mean of predictive posterior
    xhat = mean(xpost, 3);

    quantl = quantile(xpost, 0.025, 3);
    quantu = quantile(xpost, 0.975, 3);
    lower = [t; flipud(t)];
    upper = [quantl, fliplr(quantu)];

    postColor = [0.3010 0.7450 0.9330];
    meanColor = 'b';
    truthColor = 'k';
    for i = 1:n
        figure
        post = fill(lower, upper(i,:), postColor, 'FaceAlpha', 0.05);
        hold on
        data = scatter(tdata, y(i,:), 'filled', 'k');
        for j = 1:100
            samp = plot(t, xpost(i,:,j), 'Color', postColor);
            samp.Color(4) = 0.05;
        end
        est = plot(t, xhat(i,:), 'Color', meanColor);
        hold on
        truth = plot(t, x(i,:),'--', 'Color', truthColor);
        if (i==n)
          legend([post,samp,data,est,truth], 'Posterior', 'Samples',...
             'Data', 'Mean', 'Truth', 'Location', 'eastoutside');
        end
        xlabel('Time (s)')
        ylabel(strcat('$x_',num2str(i),'$'))
        xlim([t(1) t(end)])
    end
end