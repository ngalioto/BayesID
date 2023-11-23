clear; close all;
rng(1);

addpath(genpath('../../../logposterior'));
addpath(genpath('../../../nlogposterior'));
addpath('../../../filtering');
addpath('../../../sampling');
addpath('../../../utils')

%% Define constants
g = 9.81;
L = 1;
Acon = [0 1; -g/L 0];
Btrue = [0;1]; Ctrue = [1 0];
pend = @(t,x,dt)expm(Acon*dt)*x + Btrue*input(t);
x0true = zeros(2,1);

% Define our system in terms of parameters
n = 2; % state dimension
m = 1; % measurement dimension
p = 1; % input dimension
nbar = 18; % subtrajectory length
NUM_DATA = 100; % number of samples per (dt,noise) pair
tf = 20; % final time of training period

%% Define structure
% indices of learnable parameters
indx0 = 1:n; indA = (1:n^2)+indx0(end); indB = (1:n*p)+indA(end);
indC = (1:m*n)+indB(end); indQ = (1:n)+indC(end); indR = (1:m)+indQ(end);
pdyn = indC(end); ptot = indR(end);

x0 = readStructure(zeros(n,1),true(n,1),indx0,ptot);
P0 = readStructure(zeros(n), false(n), [], ptot);
A = readStructure(zeros(n,n),true(n,n),indA,ptot);
B = readStructure(zeros(n,p),true(n,p),indB,ptot);
C = readStructure(zeros(m,n),true(m,n),indC,ptot);
Q = readStructure(zeros(n,n),diag(true(n,1)),indQ,ptot);
R = readStructure(zeros(m,m),diag(true(m,1)),indR,ptot);
% place half normal priors on variance parameters
logprior = @(theta)formPrior(indQ,indR,theta);  

%% Pre-allocate for results
noise_ratio = 0:0.025:.2;
dt = 0.1:0.05:0.5;
Yfull = cell(length(dt),length(noise_ratio));
THETA = cell(length(dt),length(noise_ratio));
LSQ = cell(length(dt),length(noise_ratio));

%% Generate data
Ufull = cell(length(dt),length(noise_ratio));
sigmaRfull = zeros(NUM_DATA, length(dt), length(noise_ratio));
for i = 1:length(dt)
    Adis = expm(Acon*dt(i));
    T = floor(tf / dt(i));
    for j = 1:length(noise_ratio)
        Ufull{i,j} = sqrt(dt(i))*randn(NUM_DATA, 2*T+1);
        y = simulateDiscrete(x0true,Adis,Btrue,Ctrue,Ufull{i,j});
        sigmaRfull(:,i,j) = noise_ratio(j)*max(y,[],2);
        Yfull{i,j} = y + sigmaRfull(:,i,j).*randn(size(y));
    end
end

%% Perform estimation
for j = 1:length(dt) % loop through timestep
    T = floor(tf / dt(j)); % number of data
    K = T - nbar + 1; % number of sub-trajectories
    for k = 1:length(noise_ratio) % loop through noise level
        TT = zeros(ptot, NUM_DATA);
        DD = zeros(n^2+n*p+m*n, NUM_DATA);
        for iter = 1:NUM_DATA % loop through data samples
            y = Yfull{j,k}(iter,:); % extract data realization
            u = Ufull{j,k}(iter,:); % extract input realization
            sigmaR = sigmaRfull(iter,j,k); % extract noise level
            Y = formY(y,K,nbar); % form Y matrix
            U = formU(u,K,nbar); % form U matrix

            % define objective for given u,y
            objective = @(theta)-lpLinear(x0(theta).val, P0(theta).val, ...
                A(theta).val, B(theta).val, C(theta).val, [], ...
                Q(theta).val, R(theta).val, y(:,1:T), u(:,1:T),-logprior(theta).val);
            
            Ghat = Y / U; % LS estimation
            
            % ERA
            rows = ceil(nbar / 2);
            cols = nbar - rows;
            Ehat = formHankel(Ghat,m,p,rows,cols);
            [Ahat,Bhat,Chat] = ERA(Ehat,n,m,p);
            
            % optimization
            theta_init = [zeros(n,1);randn(8,1);[1e-6;1e-6;sigmaR^2+1e-8]];
            options = optimoptions('fmincon', 'MaxIterations', 1e3, 'MaxFunctionEvaluations', 1e5, 'Display', 'none');
            theta0 = fmincon(objective, theta_init, [],[],[],[],...
                [-Inf*ones(pdyn,1);zeros(ptot-pdyn,1)],[], [],options);
            
            % Store results
            TT(:,iter) = theta0;
            DD(:,iter) = [Ahat(:);Bhat(:);Chat(:)];
        end
        THETA{j,k} = TT;
        LSQ{j,k} = DD;
        fprintf('Element (%d, %d) complete\n', j,k);
    end
end

%% Compute MSE
MSE = cell(length(dt), length(noise_ratio));
MSElsq = cell(length(dt), length(noise_ratio));
MM = zeros(2,NUM_DATA); MMlsq = zeros(2,NUM_DATA);
for i = 1:length(dt)
    Adis = expm(Acon*dt(i));
    T = floor(tf/dt(i));
    xhat = zeros(n,2*T+1); xlsq = zeros(n,2*T+1);
    for j = 1:length(noise_ratio)
        ytrue = simulateDiscrete(x0true,Adis,Btrue,Ctrue,Ufull{i,j});
        TT = THETA{i,j};
        DD = LSQ{i,j};
        for iter = 1:NUM_DATA
            u = Ufull{i,j}(iter,:); % extract input realization
            y = ytrue(iter,:); % extract corresponding output
            theta = TT(:,iter); % extract MAP estimate
            lsq = DD(:,iter); % extract LS estimate
            % evaluate MAP MSE
            MM(:,iter) = evalMSE(x0(theta).val, A(theta).val, ...
                    B(theta).val, C(theta).val, u, y, T);
            % rearrange LS parameters into matrices:
            Alsq = reshape(lsq(1:n^2),[n,n]); 
            Blsq = lsq(n^2+1:n^2+n); 
            Hlsq = lsq(n^2+n+1:end)';
            % evaluate LS MSE
            MMlsq(:,iter) = evalMSE(zeros(n,1), Alsq, Blsq, Hlsq, u, y, T);
        end
        MSE{i,j} = MM;
        MSElsq{i,j} = MMlsq;
    end
end

%% Re-run optimization for bad stopping points
for ii_DT = 1:length(dt)
    Adis = expm(Acon*dt(ii_DT));
    T = floor(tf / dt(ii_DT));
    for ii_NR = 1:length(noise_ratio)
        ii = 0; fval = 1; obj_cur = 0;
        while(abs(fval-obj_cur) > 1e-3)
            ii = ii + 1;
            [M,I] = max(MSE{ii_DT,ii_NR}(1,:)); % find worst iteration
            y = Yfull{ii_DT,ii_NR}(I,:); % extract data realization
            u = Ufull{ii_DT,ii_NR}(I,:); % extract input realization
            U = formU(u,T,K);

            objective = @(theta)-lpLinear(x0(theta).val, P0(theta).val, ...
                A(theta).val, B(theta).val, C(theta).val, [], ...
                Q(theta).val, R(theta).val, y(:,1:T), u(:,1:T),-logprior(theta).val);
            obj_cur = objective(THETA{ii_DT,ii_NR}(:,I)); % current objective value
            fval = Inf; jj = 0;
            while (obj_cur < fval && jj < 5) % try up to 5x
                theta_init = [zeros(n,1);randn(8,1);[1e-6;1e-6;sigmaRfull(I,ii_DT,ii_NR)^2+1e-8]];
                [theta0,fval,~,~,~,~,~] = fmincon(objective, theta_init, [],[],[],[],...
                        [-Inf*ones(pdyn,1);zeros(ptot-pdyn,1)],[], [],options);
                jj = jj + 1;
            end
            if (jj < 5) % Better minimum found. Update results
                THETA{ii_DT,ii_NR}(:,I) = theta0;
                ytrue = simulateDiscrete(x0true,Adis,Btrue,Ctrue,u);
                MSE{ii_DT,ii_NR}(:,I) = evalMSE(x0(theta0).val, A(theta0).val, ...
                    B(theta0).val, C(theta0).val, u, ytrue, T);
            else
                fval = obj_cur;
            end
        end
        fprintf('(%d, %d) finished\n', ii_DT, ii_NR);
    end
end

%% Sampling
%%% iter=21 for high-noise; iter=84 for low-noise %%%
% Choose data realization
ii_dt = 1;
ii_noise = 9;
iter = 21;

T = tf / dt(ii_dt);
y = Yfull{ii_dt, ii_noise}(iter,:); u = Ufull{ii_dt, ii_noise}(iter,:);

% sampling parameters
nGibbs = 3;
num_samp = 1e6;
propC = 1e-5*eye(ptot);
logPosterior = @(theta)lpLinear(x0(theta).val, P0(theta).val, ...
                A(theta).val, B(theta).val, C(theta).val, [], ...
                Q(theta).val, R(theta).val, y(:,1:T), u(:,1:T),-logprior(theta).val);

thetafix = THETA{ii_dt,ii_noise}(:,iter); % fix obsveration parameters
% Define conditional posteriors
obj = cell(nGibbs,1);
obj{1} = @(x,theta)logPosterior([x;theta(3:6);thetafix(7:10);theta(7:9)]);
obj{2} = @(x,theta)logPosterior([theta(1:2);x;thetafix(7:10);theta(7:9)]);
obj{3} = @(x,theta)logPosterior([theta(1:6);thetafix(7:10);x]);

Cgibbs = {propC(1:2,1:2), propC(3:6,3:6), propC(11:13,11:13)};
theta0 = thetafix([1:6,11:13]);
idx = {1:2, 3:6, 7:9};
        
[samples, accR] = Gibbs_DRAM(idx,theta0,Cgibbs,num_samp,obj);

function MSE = evalMSE(x0,A,B,C,u,y,T)
    yhat = simulateDiscrete(x0,A,B,C,u);
    res = y - yhat;
    MSE = zeros(2,1);
    MSE(1) = mean(res(2:T+1).^2);
    MSE(2) = mean(res(T+2:end).^2);
end

function lp = formPrior(indQ,indR,theta)
    numParam = length(theta);
    lp.grad = zeros(numParam,1);
    Q = theta(indQ); R = theta(indR);
    if (sum(Q<0) == 0 && sum(R<0)==0)
        lp.val = 0.5*(1e6*(Q'*Q) + (R'*R));
        lp.grad([indQ,indR]) = [1e6*Q;R];
    else
        lp.val = Inf;
    end
end

% vectorized for multiple path realizations at once
function Y = simulateDiscrete(x0,A,B,C,U)
    % for 1D inputs and outputs
    [N,T] = size(U);
    X = zeros(2,N*(T));
    idx = 1:N;
    X(:,idx) = kron(ones(1,N),x0);
    for i = 1:T-1
        X(:,idx+N) = A*X(:,idx) + B*reshape(U(:,i), [1,N]);
        idx = idx + N;
    end
    Y = reshape(C*X, [N,T]);
end

function Ehat = formHankel(Ghat,m,p,rows,cols)
    Ehat = zeros(m*rows,p*cols);
    for i = 1:rows
        Ehat((1:m)*i,:) = Ghat(:,(1:cols*p) + p*(i-1));
    end
end

function [A,B,H] = ERA(E,r,m,p)
    [U,S,V] = svd(E(:,1:end-p));
    
    Ur = U(:, 1:r);
    Sr = S(1:r, 1:r);
    sqrtS = sqrt(Sr);
    Vr = V(:, 1:r);
    
    O = Ur * sqrtS;
    C = sqrtS * Vr';
    
    A = O \ E(:,(p+1):end) / C;
    B = C(:, 1:p);
    H = O(1:m, :);
end