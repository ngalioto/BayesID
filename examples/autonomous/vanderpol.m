clear; close all;
addpath('utils'); addpath('utils/ukf');
rng(1);

% UKF parameters
alpha = 1e-3;
beta = 2;
kappa = 0;
eps = 1e-10;

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
polyorder = 3;
usesine = 0;
x0 = [0;2];
P0 = 1e-16*eye(n); % known initial condition, but must be non-zero for UKF
pdyn = getNumParam(n,polyorder,usesine); %number of dynamics parameters
pvar = 3; %number of variance parameters
Q = @(theta)[theta(end-2) 0; 0 theta(end-1)];

% Measurement model
sigmaR = 2.5e-1;
H = eye(n);
h = @(x,theta)H*x;
m = size(H,1);
R = @(theta)theta(end)*eye(m);
y = generateData(@(x)van(x), x0, t, H, sigmaR);

% Sampling parameters
num_samp = 2e5; %number of samples

%%
% The discrete time dynamics (\Psi in the paper)
f = @(x,theta)fwdEuler(x,@(x)(poolData(x',n,polyorder,usesine)*...
    reshape(theta(1:end-pvar), [pdyn/n,n]))',dt/DT,DT);

% Log of the prior distribution
lambda = 0.1; %sparsity knob
logprior = @(theta)log(rhnpdf(theta(pdyn+1:end),zeros(pvar,1),eye(pvar))) - ...
    lambda*norm(theta(1:end-pvar),1);

% Optimization
objective = @(theta)-ukflp(theta, x0, P0, @(x)f(x,theta), h, ...
    Q(theta), R(theta), y, logprior,alpha,beta,kappa,eps);
theta_init = [zeros(pdyn,1); 0.1*ones(pvar,1)]; %anywhere where objective is defined
options = optimoptions('fmincon', 'MaxIterations', 100);
[theta0,~,~,~,~,~,~] = fmincon(objective, theta_init, [],[],[],[],...
    [-Inf*ones(pdyn,1);zeros(pvar,1)],[], [],options);

% Compute the Hessian
[theta0, ~,~,~,~, hessian] = fminunc(objective, theta0);

% Condition Hessian to ensure positive definiteness
propC = condHess(hessian);

%% Sample from posterior
[samples, acc] = BayesNLin(y, theta0, propC*1e-12, num_samp, x0, P0,...
    f, h, Q, R, logprior);

%%
[t_fine, x] = ode45(@(t,x)van(x), 0:dt_fine:2*tf, x0);
x = x';

xdot = @(x,theta)(poolData(x',n,polyorder,usesine)*reshape(theta(1:pdyn), [pdyn/n,n]))';
N = 100;
ind = 5e4 / N;
xpost = zeros(n,length(t_fine),N);
for i = 1:N
    [~, xhat] = ode45(@(t,x)xdot(x,samples(:,i*ind+5e4)), t_fine, x0);
    xpost(:,:,i) = xhat';
end

%%
close all;
plotResults(t_fine, x0, x, t(2:end), y, xpost);


function plotResults(t,x0, x, tdata, y, xpost)
    plottingPreferences();

    n = length(x0);
    T = length(t);
    N = size(xpost, 3);

    % Get mode of predictive posterior
    tmp1 = zeros(T,N);
    tmp2 = zeros(T,N);
    for i = 1:N
        tmp1(:,i) = xpost(1,:,i);
        tmp2(:,i) = xpost(2,:,i);
    end
    xhat(:,1) = getMode(tmp1);
    xhat(:,2) = getMode(tmp2);

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
    hold on
    for i = 1:N
        post = plot(xpost(1,:,i), xpost(2,:,i), 'Color', postColor);
        post.Color(4) = 0.05;
    end
    bayes = plot(xhat(:,1), xhat(:,2), 'Color', modeColor);
    truth = plot(x(1,:), x(2,:), '--', 'Color', truthColor);
    legend([post,bayes,truth],'Posterior', 'Mode', 'True', 'Location', 'best');
    xlabel('$x_1$')
    ylabel('$x_2$')
end