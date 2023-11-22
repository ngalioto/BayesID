clear; close all;
addpath('../../utils');
addpath(genpath('../../logposterior'));
addpath(genpath('../../nlogposterior'));
addpath(genpath('../../filtering'));
addpath(genpath('../../sampling'));
warning('off','MATLAB:illConditionedMatrix');
warning('off','MATLAB:nearlySingularMatrix');

% Lorenz system
ptrue = [10; 8/3; 28];

% Time parameters
tf = 10;
T = 100;
dt = tf / T; % should be multiple of DT
t = 0:dt:tf;
dt_fine = 0.01;
DT = 0.01; %for forward euler
if (dt < DT)
    DT = dt;
end

% Dynamics model
n = 3;
m = 3;
pdyn = 3; %number of dynamics parameters
pvar = 4; %number of variance parameters
ptot = pdyn+pvar;
x0true = [2.0181; 3.5065; 11.8044];
P0true = 1e-16*eye(n); %must be non-zero for UKF
Htrue = eye(n);

indQ = pdyn+1:pdyn+n; indR = ptot;
x0 = readStructure(x0true,false(n,1),[]); %fixed
P0 = readStructure(P0true,false(n),[]); %fixed
f = @(x,theta)fwdEuler(x,@(x)Lorenz(x,theta),dt/DT,DT); %learnable (\Psi in the paper)
H = readStructure(Htrue,false(n),[]); %fixed
Q = readStructure(zeros(n),diag(true(n,1)),indQ); %learnable
R = readStructure(zeros(m),diag(true(m,1)),indR); %learnable

% Generate data
sigmaR = 2;
y = generateData(@(x)Lorenz(x,ptrue), x0true, t, Htrue, sigmaR);

% UKF parameters
alpha = 1e-3;
beta = 2;
kappa = 0;
epsilon = 1e-10;
lambda = alpha^2 * (n+kappa) - n;
[Wm, Wc] = formWeights(n, lambda, alpha, beta);

%%
% Log of the prior distribution
logprior = @(theta)formPrior(theta,pdyn+1:ptot);

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
num_samp = 1e4; %number of samples
evalLogPost = @(theta)lpNLDyn(x0(theta).val, P0(theta).val, @(x)f(x,theta),...
    H(theta).val, [], Q(theta).val, R(theta).val, [], y,...
    logprior(theta).val, lambda, Wm, Wc);
[samples, acc] = DRAM(theta0,C0,num_samp,evalLogPost);

%%
[t_fine, x] = ode45(@(t,x)Lorenz(x,ptrue), 0:dt_fine:2*tf, x0true);
x = x';

N = 100;
ind = 5e3 / N;
xpost = zeros(n,length(t_fine),N);
for i = 1:N
    [~, xhat] = ode45(@(t,x)Lorenz(x,samples(:,i*ind+5e3)), t_fine, x0true);
    xpost(:,:,i) = xhat';
end

%%
close all;
plotResults(t_fine, @(x)Lorenz(x,theta0),x0true, x, t, y, xpost);


function xdot = Lorenz(x, p)
    xdot(1,1) = p(1)*(x(2) - x(1));
    xdot(2,1) = x(1)*(p(3) - x(3)) - x(2);
    xdot(3,1) = x(1)*x(2) - p(2)*x(3);
end

function lp = formPrior(theta,indVar)
    pvar = theta(indVar);
    if (any(pvar < 0))
        lp.val = -Inf;
        lp.grad = NaN*ones(length(theta),1);
    else
        lp.val = -0.5*(pvar'*pvar);
        lp.grad = zeros(length(theta),1);
        lp.grad(indVar) = -pvar;
    end
end

function plotResults(t, f, x0, x, tdata, y, xpost)
    plottingPreferences();

    n = length(x0);
    T = length(t);
    N = size(xpost, 3);

    % Simulate with mode from parameter posterior
    [~, xhat] = ode45(@(t,x)f(x),t,x0);

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
        est = plot(t, xhat(:,i), 'Color', modeColor);
        hold on
        truth = plot(t, x(i,:),'--', 'Color', truthColor);
        if (i==n)
          legend([post,samp,data,est,truth], 'Posterior', 'Samples',...
             'Data', 'Mode', 'Truth', 'Location', 'eastoutside');
        end
        xlabel('Time (s)')
        ylabel(strcat('$x_',num2str(i),'$'))
    end
    
    figure
    postColor = [0.1059 0.6196 0.4667];
    subplot(1,2,1);
    hold on;
    for i = 1:50
        post = plot3(xpost(1,:,i*(N/50)),xpost(2,:,i*(N/50)),xpost(3,:,i*(N/50)),...
            'Color', postColor);
        post.Color(4) = 0.05;
    end
    view(47.6151, 10.1341)
    xlabel('$x$');ylabel('$y$');zlabel('$z$');
    title('Posterior')
    subplot(1,2,2);
    truth = plot3(x(1,:), x(2,:), x(3,:), 'Color', postColor);
    view(47.6151, 10.1341)
    xlabel('$x$');ylabel('$y$');zlabel('$z$');
    title('Truth')
end

function plottingPreferences()
    N = 16;
    set(0,'DefaultLineLineWidth',2)
    
    set(0,'defaultAxesFontSize',N)
    set(0, 'defaultLegendFontSize', N)
    set(0, 'defaultColorbarFontSize', N);
    
    set(0,'defaulttextinterpreter','latex')
end