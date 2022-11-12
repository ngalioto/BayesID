clear; close all;
addpath('../../utils');
addpath(genpath('../../logposterior'));
addpath(genpath('../../nlogposterior'));
addpath(genpath('../../filtering'));
addpath(genpath('../../sampling'));
rng(1);

% Time parameters
tf = 4;
T = 40;
dt = tf / T;
t = 0:dt:tf;
dt_fine = 0.01; % for data generation/simulating truth

% Dynamics model
n = 2;
m = 2;
x0true = [0.1; -0.5];
indA = 1:n^2; indQ = n^2+1; indR = n^2+2;
pdyn = indA(end); ptot = indR(end);
x0 = readStructure(x0true,false(n,1),[],ptot);
P0 = readStructure(zeros(n),false(n),[],ptot);
A = readStructure(zeros(n),true(n),1:pdyn,ptot);
H = readStructure(eye(n),false(n),[],ptot);
Q = readStructure(zeros(n),diag(true(n,1)),pdyn+1,ptot);
R = readStructure(zeros(m),diag(true(m,1)),ptot,ptot);

% Pendulum system and initial conditions
g = 9.81;
L = 1;
Acon = [0 1; -g/L 0];
ptrue = reshape(expm(Acon*dt)', [pdyn, 1]);
pend = @(x)Acon*x;
% pend = @(x)[x(2); -g/L*sin(x(1))]; % nonlinear pendulum

% Generate data
sigmaR = 0.1;
y = generateData(@(x)pend(x), x0true, t, H(ptrue).val, sigmaR);

% Sampling parameters
num_samp = 1e3;

%%
% Log of the prior distribution
lambda = 0.1; %sparsity knob
nlogprior = @(theta)formPrior(theta,pdyn+1:ptot);
% logprior = @(theta)log(rhnpdf(theta(pdyn+1:end),zeros(pvar,1),eye(pvar))) - ...
%     lambda*norm(theta(1:pdynd),1);

% Optimization
objective = @(theta)nlpLinear(1:ptot, x0(theta), P0(theta), A(theta), [], ...
    H(theta), [], Q(theta), R(theta), [], y, nlogprior(theta));
% objective = @(theta)-kflp(theta, x0, P0, A(theta), H(theta), ...
%     Q(theta), R(theta), y, logprior, eye(m));
theta_init = [zeros(pdyn,1); 0.1*ones(ptot-pdyn,1)]; %anywhere where objective is defined
options = optimoptions('fmincon', 'SpecifyObjectiveGradient', true, 'Display', 'Iter');
[theta0,~,~,~,~,~,hessian] = fmincon(objective, theta_init, [],[],[],[],...
    [-Inf*ones(pdyn,1);zeros(ptot-pdyn,1)],[], [],options);

% Condition Hessian to ensure positive definiteness
C0 = conditionHess(hessian);

%% Sample from posterior
evalLogPost = @(theta)lpLinear(x0(theta).val, P0(theta).val, A(theta).val, [], ...
    H(theta).val, [], Q(theta).val, R(theta).val, [], y, -nlogprior(theta).val);
[samples, acc] = DRAM(theta0,C0,num_samp,evalLogPost);
% [samples, acc] = BayesLin(y, theta0, propC, num_samp, x0, P0,...
%     A, H, Q, R, logprior);

%%
[t_fine, x] = ode45(@(t,x)pend(x), 0:dt_fine:2*tf, x0);
x = x';

Acon = @(theta)logm(A(theta)) / dt;
N = 100;
ind = num_samp / N;
xpost = zeros(n,length(t_fine),N);
for i = 1:N
    [~, xhat] = ode45(@(t,x)Acon(samples(:,i*ind))*x, t_fine, x0);
    xpost(:,:,i) = xhat';
end

%%
close all;
plotResults(t_fine, x0, x, t(2:end), y, xpost);


function nlp = formPrior(theta,indVar)
    pvar = theta(indVar);
    if (any(pvar < 0))
        nlp.val = Inf;
        nlp.grad = NaN*ones(length(theta),1);
    else
        nlp.val = 0.5*(pvar'*pvar);
        nlp.grad = zeros(length(theta),1);
        nlp.grad(indVar) = pvar;
    end
end

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

function propC = conditionHess(hessian)
    n = size(hessian,1);
    for i = 1:n^2
        if (isnan(hessian(i)))
            hessian(i) = 1e10;
        elseif hessian(i) == 0
            hessian(i) = 1e-8;
        end
    end
    i = -16;
    while (sum(eig(inv(hessian) + 10^i*eye(n)) <= 0) > 0)
        i = i + 1;
    end
    propC = inv(hessian) + 10^i*eye(n);
end