clear; close all;
addpath('../../utils');
addpath(genpath('../../logposterior'));
addpath(genpath('../../nlogposterior'));
addpath(genpath('../../filtering'));
addpath(genpath('../../sampling'));
warning('off','MATLAB:illConditionedMatrix');
warning('off','MATLAB:nearlySingularMatrix');

% Van der Pol system
mu = 3;
van = @(x)[x(2);mu*(1-x(1)^2)*x(2)-x(1)];

% Time parameters
tf = 20;
T = 200;
dt = tf / T; % should be multiple of DT
t = 0:dt:tf;
dt_fine = 0.01;
DT = 0.01; %for forward euler
if (dt < DT)
    DT = dt;
end

% Dynamics model
n = 2;
m = 2;
polyorder = 3;
usesine = 0;
pdyn = getNumParam(n,polyorder,usesine); %number of dynamics parameters
pvar = 3; %number of variance parameters
ptot = pdyn+pvar;
x0true = [0;2];
P0true = 1e-16*eye(n); % known initial condition, but must be non-zero for UKF
Htrue = eye(n);

indQ = pdyn+1:pdyn+n; indR = ptot;
x0 = readStructure(x0true,false(n,1),[]); %fixed
P0 = readStructure(P0true,false(n),[]); %fixed
% The discrete time dynamics (\Psi in the paper)
f = @(x,theta)fwdEuler(x,@(x)(poolData(x',n,polyorder,usesine)*...
    reshape(theta(1:end-pvar), [pdyn/n,n]))',dt/DT,DT);
H = readStructure(Htrue,false(n),[]); %fixed
Q = readStructure(zeros(n),diag(true(n,1)),indQ); %learnable
R = readStructure(zeros(m),diag(true(m,1)),indR); %learnable

% Generate data
sigmaR = 2.5e-1;
y = generateData(@(x)van(x), x0true, t, Htrue, sigmaR);

% UKF parameters
alpha = 1e-3;
beta = 2;
kappa = 0;
epsilon = 1e-10;
lambda = alpha^2 * (n+kappa) - n;
[Wm, Wc] = formWeights(n, lambda, alpha, beta);

%%
% Log of the prior distribution
logprior = @(theta)formPrior(theta,1:pdyn,pdyn+1:ptot);

% Optimization
objective = @(theta)-lpNLDyn(x0(theta).val, P0(theta).val, ...
    @(x)f(x,theta), H(theta).val, [], Q(theta).val, R(theta).val, [], y, ...
    logprior(theta).val, lambda, Wm, Wc);
theta_init = [zeros(pdyn,1); 0.1*ones(pvar,1)]; %anywhere where objective is defined
options = optimoptions('fmincon', 'Display', 'Iter');
[theta0,~,~,~,~,~,hessian] = fmincon(objective, theta_init, [],[],[],[],...
    [-Inf*ones(pdyn,1);zeros(pvar,1)],[], [],options);

% Condition Hessian to ensure positive definiteness
C0 = conditionHess(hessian);

%% Sample from posterior
num_samp = 1e5; %number of samples
evalLogPost = @(theta)lpNLDyn(x0(theta).val, P0(theta).val, @(x)f(x,theta),...
    H(theta).val, [], Q(theta).val, R(theta).val, [], y,...
    logprior(theta).val, lambda, Wm, Wc);
[samples, acc] = DRAM(theta0,C0,num_samp,evalLogPost);

%%
[t_fine, x] = ode45(@(t,x)van(x), 0:dt_fine:2*tf, x0true);
x = x';

burn_in = 5e4;
xdot = @(x,theta)(poolData(x',n,polyorder,usesine)*reshape(theta(1:pdyn), [pdyn/n,n]))';
N = 100;
ind = burn_in / N;
xpost = zeros(n,length(t_fine),N);
for i = 1:N
    theta = samples(:,i*ind+burn_in);
    [~, xhat] = ode45(@(t,x)xdot(x,theta), t_fine, x0(theta).val);
    xpost(:,:,i) = xhat';
end
[~, xmap] = ode45(@(t,x)xdot(x,theta0), t_fine, x0(theta0).val);

%%
close all;
plotResults(t_fine, x0true, x, t, y, xpost, xmap);

function lp = formPrior(theta,indDyn,indVar)
    pvar = theta(indVar);
    if (any(pvar < 0))
        lp.val = -Inf;
        lp.grad = NaN*ones(length(theta),1);
    else
        lp.val = -0.5*(pvar'*pvar) - 0.1*norm(theta(indDyn,1));
        lp.grad = zeros(length(theta),1);
        lp.grad(indDyn) = -sign(theta(indDyn));
        lp.grad(indVar) = -pvar;
    end
end

function plotResults(t,x0, x, tdata, y, xpost, xmap)
    plottingPreferences();

    n = length(x0);
    N = size(xpost, 3);
    
    quantl = quantile(xpost, 0.025, 3);
    quantu = quantile(xpost, 0.975, 3);
    lower = [t; flipud(t)];
    upper = [quantl, fliplr(quantu)];

    postColor = [0.3010 0.7450 0.9330];
    modeColor = 'b';
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
        est = plot(t, xmap(:,i), 'Color', modeColor);
        hold on
        truth = plot(t, x(i,:),'--', 'Color', truthColor);
        if (i==n)
          legend([post,samp,data,est,truth], 'Posterior', 'Samples',...
             'Data', 'MAP', 'Truth', 'Location', 'eastoutside');
        end
        xlabel('Time (s)')
        ylabel(strcat('$x_',num2str(i),'$'))
    end
    
    figure
    postColor = [0.1059 0.6196 0.4667];
    hold on
    for i = 1:N
        post = plot(xpost(1,:,i), xpost(2,:,i), 'Color', postColor);
        post.Color(4) = 0.05;
    end
    bayes = plot(xmap(:,1), xmap(:,2), 'Color', modeColor);
    truth = plot(x(1,:), x(2,:), '--', 'Color', truthColor);
    legend([post,bayes,truth],'Posterior', 'MAP', 'True', 'Location', 'best');
    xlabel('$x_1$')
    ylabel('$x_2$')
end

function plottingPreferences()
    N = 16;
    set(0,'DefaultLineLineWidth',2)
    
    set(0,'defaultAxesFontSize',N)
    set(0, 'defaultLegendFontSize', N)
    set(0, 'defaultColorbarFontSize', N);
    
    set(0,'defaulttextinterpreter','latex')
end