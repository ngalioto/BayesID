addpath(genpath('../../../logposterior'));
addpath(genpath('../../../nlogposterior'));
addpath('../../../filtering');
addpath('../../../sampling');
addpath('../../../utils');

clear; clc; close all;

%% Constants
n = 8;
m = 1;
p = 1;

num_nodes = 15;
sigmaR = 2e-1;
dt = 1e-1; T = 100;
Htrue = [1,zeros(1,n-1)];

%% Load and process data
T0 = 200; % remove transient behavior
TF = 400;
num_spatial_nodes = 257;
load('acSolution.mat');

% Raw training data
uraw = ux(:,T0:T0+T);
yraw = (sol(:,T0:T0+T) + sigmaR*randn(m,T+1) / num_spatial_nodes);
yscale = sqrt(var(yraw(:))); yshift = mean(yraw(:));
uscale = sqrt(var(uraw(:))); ushift = mean(uraw(:));

% Processed training data
y = (yraw - yshift) ./ yscale;
u = (ux(:,T0:TF) - ushift) ./ uscale;

% Testing data (clean)
y_test = sol(:,T0:TF) / num_spatial_nodes;

%% Define our system in terms of parameters
f_params = n*(n+p+1+num_nodes) + num_nodes*(n+p+1);
indx0 = 1:n; indF = (1:f_params)+indx0(end);
indQ = (1:n)+indF(end); indR = (1:m)+indQ(end);
pdyn = indF(end); ptot = indR(end); pvar = ptot-pdyn;

x0 = readStructure(zeros(n,1),true(n,1),indx0,ptot);
P0 = readStructure(1e-8*eye(n), false(n), [], ptot);

f = @(idx,x,u,theta)simple_rnn_grad(idx, struct('val',[x.val;u.*ones(p,size(x.val,2))],...
    'grad',x.grad),ptot,theta,indF(1),n,n+p,n,num_nodes);
H = readStructure(Htrue,false(m,n),[],ptot);

Q = readStructure(zeros(n,n),diag(true(n,1)),indQ,ptot);
R = readStructure(zeros(m,m),diag(true(m,1)),indR,ptot);

prior = 'DynQR';
Qmean = zeros(n,1); Qinv = 1e6*eye(n);
Rmean = zeros(m,1); Rinv = 1e0*eye(m);
nlp = @(theta)assembleNLP(theta,ptot,pdyn,indQ,Qmean,Qinv,indR,Rmean,Rinv,prior);

%% Define UKF and negative log posterior
% UKF parameters
alpha = 1e-3;
beta = 2;
kappa = 0;
eps = 1e-10;

lambda = alpha^2 * (n+kappa) - n;
[Wm, Wc] = formWeights(n, lambda, alpha, beta);
evalNLP = @(theta)nlpNLDyn(1:ptot, x0(theta), P0(theta), @(idx,x,u)f(idx,x,u,theta),...
    H(theta), [], Q(theta), R(theta), u, y, nlp(theta),lambda,Wm,Wc);

%% Optimization
theta_init = [1e-2*randn(pdyn,1);abs(mvnrnd(Qmean,inv(Qinv)))';...
    abs(mvnrnd(Rmean,inv(Rinv)))'];
opt_options = optimoptions('fmincon','SpecifyObjectiveGradient',true,...
    'Display','iter','MaxIterations',1e3,'MaxFunctionEvaluations',1e7);
theta_map = fmincon(evalNLP,theta_init,[],[],[],[],...
    [-Inf*ones(pdyn,1);zeros(pvar,1)],[],[],opt_options);

%% Deterministic least squares
fvec = @(x,u,theta)simple_res_net([x;u.*ones(p,size(x,2))],...
    theta(indF),n+p,n,num_nodes); % f without grad computation
opt = optimoptions('fmincon','Display','iter','MaxIterations',1e3,'MaxFunctionEvaluations',1e7);
yhat = @(theta)simulate(x0(theta).val,@(x,u)fvec(x,u,theta),@(x,u)H(theta).val*x,u,T+1);
obj_ls = @(theta)mean((y - yhat(theta)).^2);
theta_ls = fmincon(obj_ls,theta_map(1:pdyn),[],[],[],[],-Inf*ones(pdyn,1),[],[],opt);

%% Sampling
idx = {indx0,indF,[indQ,indR]};
logPosterior = @(theta)lpNLDyn(x0(theta).val, P0(theta).val, ...
    @(x,u)fvec(x,u,theta), H(theta).val, [], Q(theta).val, R(theta).val, ...
    u, y, -nlp(theta).val,lambda,Wm,Wc);
obj = cell(3,1);
obj{1} = @(x,theta)logPosterior([x;theta(indF);theta((indF(end)+1):end)]);
obj{2} = @(x,theta)logPosterior([theta(indx0);x;theta((indF(end)+1):end)]);
obj{3} = @(x,theta)logPosterior([theta([indx0,indF]);x]);
propGibbs = {1e-8*eye(n), 1e-6*eye(length(indF)), 1e-8*eye(pvar)};
M = 1e5;
[samples,accR,propC] = Gibbs_DRAM(idx,theta_map,propGibbs,M,obj);

function nlp = assembleNLP(theta,ptot,pdyn,indQ,Qmean,Qinv,indR,Rmean,Rinv,prior)
    nlp.val = 0;
    nlp.grad = zeros(ptot,1);
    if (sum(theta([indQ,indR])<0) > 0)
        nlp.val = Inf;
        nlp.grad = nan*ones(ptot,1);
    else
        if (contains(prior,'Dyn'))
            nlp.val = nlp.val + 2*theta(1:pdyn)'*theta(1:pdyn);
            nlp.grad(1:pdyn) = nlp.grad(1:pdyn) + 4*theta(1:pdyn);
        end
        if (contains(prior,'Q'))
            resQ = theta(indQ)-Qmean;
            nlp.val = nlp.val + resQ'*Qinv*resQ;
            nlp.grad(indQ) = nlp.grad(indQ) + 2*Qinv*resQ;
        end
        if (contains(prior,'R'))
            resR = theta(indR)-Rmean;
            nlp.val = nlp.val + resR'*Rinv*resR;
            nlp.grad(indR) = nlp.grad(indR) + 2*Rinv*resR;
        end
    end
end