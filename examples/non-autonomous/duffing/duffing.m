addpath(genpath('../../../logposterior'));
addpath(genpath('../../../nlogposterior'));
addpath('../../../filtering');
addpath('../../../sampling');
addpath('../../../utils')

clear; clc; close all;

rng(1);

%% Specify our truth model
alpha = -1;
beta = 1;
delta = 0.3;
gamma = .5;
omega = 1.2;
f = @(t,x)[x(2); -alpha*x(1) - delta*x(2) - beta*x(1)^3 + gamma*cos(omega*t)];

%% Generate data
tf = 300;
T = 1200;
dt = tf / T;
t = 0:dt:tf*4;

y0 = zeros(2,1);
Htrue = [1 0];
sigmaR = 1e-3;
[x,y] = acquireData(@(t,x)f(t,x), y0, t, Htrue, sigmaR);

%% Remove transient portion
x = x(:,end-2*T+1:end); y = y(:,end-2*T+1:end-T); tdata = t(1:T);
u = gamma*cos(omega*t(end-2*T+1:end));
t = 0:dt:tf*2;

%% Constants
n = 2; % state dimension
m = 1; % measurement dimension
p = 1; % input dimension
num_nodes = 15; % nodes in NN hidden layer

%% Define our system in terms of parameters
f_params = n*(n+p+1+num_nodes) + num_nodes*(n+p+1);
indx0 = 1:n; indF = (1:f_params)+indx0(end);
indQ = (1:n)+indF(end);
pdyn = indF(end); ptot = indQ(end); pvar = ptot-pdyn;

x0 = readStructure(zeros(n,1),true(n,1),indx0,ptot);
P0 = readStructure(1e-8*eye(n), false(n), [], ptot);

f = @(idx, x,u,theta)simple_rnn_grad(idx, struct('val',[x.val;u.*ones(p,size(x.val,2))],...
    'grad',x.grad),ptot,theta,indF(1),n,n+p,n,num_nodes);
H = readStructure(Htrue,false(m,n),[],ptot);

Q = readStructure(zeros(n,n),diag(true(n,1)),indQ,ptot);
R = readStructure(sigmaR^2*eye(m),false(m),[],ptot);

Qmean = zeros(n,1); Qinv = 1e4*eye(n);
nlp = @(theta)assembleNLP(theta,ptot,pdyn,indQ,Qmean,Qinv);

%% Define UKF and negative log posterior
% UKF parameters
alpha_UKF = 1e-3;
beta_UKF = 2;
kappa = 0;

lambda = alpha_UKF^2 * (n+kappa) - n;
[Wm, Wc] = formWeights(n, lambda, alpha_UKF, beta_UKF);
evalNLP = @(idx,theta)nlpNLDyn(idx,x0(theta), P0(theta), ...
    @(idx,x,u)f(idx,x,u,theta), H(theta), [], Q(theta), R(theta), u, ...
    y, nlp(theta),lambda,Wm,Wc);

%% Optimization
theta_init = [1e-2*randn(pdyn,1);abs(mvnrnd(Qmean,inv(Qinv)))'];
opt_map = optimoptions('fmincon','SpecifyObjectiveGradient',true,...
    'Display','iter','MaxIterations',1e2,'MaxFunctionEvaluations',1e7);
theta_map = fmincon(@(theta)evalNLP(1:ptot,theta),theta_init,...
    [],[],[],[],[-Inf*ones(pdyn,1);zeros(pvar,1)],[],[],opt_map);

%% Deterministic least squares
fvec = @(x,u,theta)simple_res_net([x;u.*ones(p,size(x,2))],...
    theta(indF),n+p,n,num_nodes); % f without gradient
yhat = @(theta)simulate(x0(theta).val,@(x,u)fvec(x,u,theta),...
                @(x,u)H(theta).val*x,u(1:T),T);
obj_ls = @(theta)mean((y-yhat(theta)).^2);
opt_ls = optimoptions('fminunc','Display','iter','MaxIterations',1e5,'MaxFunctionEvaluations',1e5);
theta_ls = fminunc(obj_ls, theta_init(1:pdyn),opt_ls);

%% Multiple shooting
msT = 200; % time parameter
numIC = ceil(size(y,2) / msT); % number initial conditions to learn
theta_ms_init = [theta_map(1:pdyn);randn(n*numIC,1)];
opt_ms = optimoptions('fmincon','Display','iter','MaxIterations',1e5,'MaxFunctionEvaluations',1e7);
obj_ms = @(theta)multipleShooting(theta(1:pdyn),...
    reshape(theta(pdyn+1:end),[n,numIC]),fvec,H,y,u,msT);
theta_ms = fmincon(obj_ms,theta_ms_init,[],[],[],[],-Inf*ones(pdyn+n*numIC,1),[],[],opt_ms);

%% Sampling
idx = {indx0,indF,indQ};
logPosterior = @(theta)lpNLDyn(x0(theta).val, P0(theta).val, ...
    @(x,u)fvec(x,u,theta), H(theta).val, [], Q(theta).val, R(theta).val, ...
    u, y, -nlp(theta).val,lambda,Wm,Wc);
% Define conditional posteriors
obj = cell(3,1);
obj{1} = @(x,theta)logPosterior([x;theta(indF);theta((indF(end)+1):end)]);
obj{2} = @(x,theta)logPosterior([theta(indx0);x;theta((indF(end)+1):end)]);
obj{3} = @(x,theta)logPosterior([theta([indx0,indF]);x]);

propGibbs = {1e-8*eye(n), 1e-8*eye(length(indF)), 1e-8*eye(pvar)};
M = 1e6;
[samples,accR,propC] = Gibbs_DRAM(idx,theta0,propGibbs,M,obj);

%% Helper functions
function obj = multipleShooting(theta,x0,f,H,y,u,T)
    fvec = @(x,u)f(x,u,theta);
    hvec = @(x,u)H(theta).val*x;

    n = size(y,2);
    numIC = ceil(n/T);
    obj = 0;
    ind = 1:T;
    for i = 1:numIC
        if (ind(end) > n)
            ind = ind(1):n;
        end
        yhat = simulate(x0(:,i), fvec, hvec, u(:,ind(1:end-1)), T);
        obj = obj + sum((y(:,ind) - yhat).^2);
        ind = ind + T;
    end
end

function [x,y]  = acquireData(f, x0, t, H, sigmaR)
    [~, x] = ode45(@(t,x)f(t,x), t, x0);
    x = x';
    y = H*x + sigmaR*randn(size(H*x));
end

function nlp = assembleNLP(theta,ptot,pdyn,indQ,Qmean,Qinv)
    nlp.val = 0;
    nlp.grad = zeros(ptot,1);

    nlp.val = nlp.val + 0.1*theta(1:pdyn)'*theta(1:pdyn);
    nlp.grad(1:pdyn) = 0.2*theta(1:pdyn);

    if (sum(theta(indQ)<0)>0)
        nlp.val = Inf;
        nlp.grad(indQ) = nan;
    else
        resQ = theta(indQ)-Qmean;
        nlp.val = nlp.val + 0.5*resQ'*Qinv*resQ;
        nlp.grad(indQ) = Qinv*resQ;
    end
end