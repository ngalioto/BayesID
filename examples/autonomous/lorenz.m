clear; close all;
addpath('utils'); addpath('utils/ukf');
rng(1);

% UKF parameters
alpha = 1e-3;
beta = 2;
kappa = 0;
eps = 1e-10;

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
x0 = [2.0181; 3.5065; 11.8044];
P0 = 1e-16*eye(n); % known initial condition, but must be non-zero for UKF
pdyn = 3; %number of dynamics parameters
pvar = 4; %number of variance parameters
Q = @(theta)[theta(end-3) 0 0; 0 theta(end-2) 0; 0 0 theta(end-1)];

% Measurement model
sigmaR = 2;
H = eye(n);
h = @(x,theta)H*x;
m = size(H,1);
R = @(theta)theta(end)*eye(m);
y = generateData(@(x)Lorenz(x,ptrue), x0, t, H, sigmaR);

% Sampling parameters
num_samp = 1e4; %number of samples

%%
% The discrete time dynamics (\Psi in the paper)
f = @(x,theta)propf(x,@(x)Lorenz(x,theta),dt/DT,DT);

% Log of the prior distribution
logprior = @(theta)log(rhnpdf(theta(pdyn+1:end),zeros(pvar,1),eye(pvar)));

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
[samples, acc] = BayesNLin(y, theta0, propC, num_samp, x0, P0,...
    f, h, Q, R, logprior);

%%
[t_fine, x] = ode45(@(t,x)Lorenz(x,ptrue), 0:dt_fine:2*tf, x0);
x = x';

N = 100;
ind = 5e3 / N;
xpost = zeros(n,length(t_fine),N);
for i = 1:N
    [~, xhat] = ode45(@(t,x)Lorenz(x,samples(:,i*ind+5e3)), t_fine, x0);
    xpost(:,:,i) = xhat';
end

%%
close all;
plotResults(t_fine, @(x)Lorenz(x,theta0),x0, x, t(2:end), y, xpost);


function xdot = Lorenz(x, p)
    xdot(1,1) = p(1)*(x(2) - x(1));
    xdot(2,1) = x(1)*(p(3) - x(3)) - x(2);
    xdot(3,1) = x(1)*x(2) - p(2)*x(3);
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