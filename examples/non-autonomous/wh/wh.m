addpath(genpath('../../../logposterior'));
addpath(genpath('../../../nlogposterior'));
addpath('../../../filtering');
addpath('../../../sampling');
addpath('../../../utils');

clear; clc; close all;
warning('off','all');

%% Load and process data
path_to_benchmark = '../path_to_benchmark';
% Full dataset can be found at nonlinearbenchmark.org
load(strcat(path_to_benchmark, '/WienerHammerBenchMark.mat'));
% Trim data to remove transience
START = 5.2e3+1;
END = 184e3;
T = END - START;
ufull = u(START:END);
yfull = y(START:END);

% specify number of training data and add noise
numData = 1e3;
u_train = ufull(:,1:numData);
y_train = yfull(:,1:numData) + 0.0178*randn(1,numData);

% normalize the data to have mean 0 variance 1
ushift = mean(u_train); uscale = sqrt(var(u_train));
yshift = mean(y_train); yscale = sqrt(var(y_train));
u_test = (ufull - ushift) ./ uscale;
y_test = (yfull - yshift) ./ yscale;
u_train = (u_train - ushift) ./ uscale;
y_train = (y_train - yshift) ./ yscale;

%% Constants
n = 6; % state dimension
m = 1; % measurement dimension
p = 1;

x0true = zeros(n,1);
Atrue = eye(n); Htrue = [eye(m),zeros(m,n-m)];
sigmaR = 0.0178*eye(m); Qtrue = 1e-8*eye(n);

%% Define our model in terms of parameters
num_nodes = 15; % number of nodes in hidden layer of NN
f_params = n*(n+p+1+num_nodes) + num_nodes*(n+p+1);
h_params = m*(n+p+1+num_nodes) + num_nodes*(n+p+1);
indx0 = 1:n; indF = (1:f_params)+indx0(end);
indH = (1:h_params)+indF(end); indQ = (1:n)+indH(end); indR = (1:m)+indQ(end);
pdyn = indH(end); ptot = indR(end); pvar = ptot-pdyn;

x0 = readStructure(zeros(n,1),true(n,1),indx0,ptot);
P0 = readStructure(2.447e-8*eye(n), false(n), [], ptot); % 0.1% of data std dev

f = @(idx, x,u,theta)simple_rnn_grad(idx, struct('val',[x.val;u.*ones(p,size(x.val,2))],...
    'grad',x.grad),ptot,theta,indF(1),n,n+p,n,num_nodes);

h = @(idx, x,u,theta)simple_rnn_grad(idx, struct('val',[x.val;u.*ones(p,size(x.val,2))],...
    'grad',x.grad),ptot,theta,indH(1),n,n+p,m,num_nodes);

Q = getCov(n,indQ,ptot);
R = getCov(m,indR,ptot);

Qmean = log(1e-5); Qinv = 1e-2*eye(n);
Rmean = log(diag(sigmaR).^2); Rinv = 1e0*eye(m);
nlp = @(theta)prior(theta,ptot,pdyn,indQ,Qmean,Qinv,indR,Rmean,Rinv);

%% Define UKF and negative log posterior
% UKF parameters
alpha = 1e-3;
beta = 2;
kappa = 0;
eps = 1e-10;

lambda = alpha^2 * (n+kappa) - n;
[Wm, Wc] = formWeights(n, lambda, alpha, beta);

evalNLP = @(theta)nlpNonlinear(1:ptot, x0(theta), P0(theta), ...
    @(idx,x,u)f(idx,x,u,theta), @(idx,x,u)h(idx,x,u,theta), ...
    Q(theta), R(theta), u_train, y_train, nlp(theta),lambda,Wm,Wc);

%% Optimization
opt_options = optimoptions('fmincon','SpecifyObjectiveGradient',true,...
    'Display','iter','MaxIterations',1e3,'MaxFunctionEvaluations',1e7);
theta_init = [1e-1*ones(pdyn,1);Qmean*ones(n,1);0];
theta_map = fmincon(evalNLP,theta_init,[],[],[],[],[],[],[],opt_options);

%% Sampling
thetafix = theta_map;
nGibbs = 3;
num_samp = 1e5;
propC = 1e-5*eye(ptot);

fvec = @(x,u,theta)simple_res_net([x;u.*ones(p,size(x,2))],theta(indF),n+p,n,num_nodes);
hvec = @(x,u,theta)simple_res_net([x;u.*ones(p,size(x,2))],theta(indH),n+p,m,num_nodes);
logPosterior = @(theta)lpNonlinear(x0(theta).val, P0(theta).val, ...
    @(x,u)fvec(x,u,theta), @(x,u)hvec(x,u,theta), Q(theta).val, ...
    R(theta).val, u_train, y_train, -nlp(theta).val,lambda,Wm,Wc);

obj = cell(nGibbs,1);
obj{1} = @(x,theta)logPosterior([x;theta(7:264);thetafix(indH);theta(265:271)]);
obj{2} = @(x,theta)logPosterior([theta(1:6);x;thetafix(indH);theta(265:271)]);
obj{3} = @(x,theta)logPosterior([theta(1:264);thetafix(indH);x]);
Cgibbs = {propC(1:6,1:6), propC(7:264,7:264), propC(265:271,265:271)};
samp0 = thetafix([1:264,408:414]);
idx = {1:6, 7:264, 265:271};
    
[samples,accR] = Gibbs_DRAM(idx,samp0,Cgibbs,num_samp,obj);

function nlp = prior(theta,ptot,pdyn,indQ,Qmean,Qinv,indR,Rmean,Rinv)
    nlp.val = 0;
    nlp.grad = zeros(ptot,1);
    
    nlp.val = nlp.val + 0.1*theta(1:pdyn)'*theta(1:pdyn);
    nlp.grad(1:pdyn) = 0.2*theta(1:pdyn);


    resQ = theta(indQ)-Qmean;
    nlp.val = nlp.val + 0.5*resQ'*Qinv*resQ;
    nlp.grad(indQ) = Qinv*resQ;


    resR = theta(indR)-Rmean;
    nlp.val = nlp.val +  0.5*resR'*Rinv*resR;
    nlp.grad(indR) = Rinv*resR;

    nlp.grad = sparse(nlp.grad);
end

function grad = getDiagMatGrad(n,ind,mat,grad) %unconstrained variance
    for i = 1:n
        partInd = n*(ind(i)-1);
        grad(partInd + i,i) = mat(i,i);
    end
    grad = sparse(grad);
end

function Q = getCov(n,ind,ptot)
    Qmat = @(theta)diag(exp(theta(ind)));
    if (~isempty(ptot))
        grad = zeros(n*ptot,n);
        grad = @(theta)getDiagMatGrad(n,ind,Qmat(theta),grad);
        Q = @(theta)struct('val',Qmat(theta),'grad',grad(theta));
    else
        Q = @(theta)struct('val',Qmat(theta));
    end
end