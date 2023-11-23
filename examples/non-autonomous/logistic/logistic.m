addpath(genpath('../../logposterior'));
addpath('../../filtering');
addpath('../../plotting');
addpath('../../utils');
clear;

%% Generate data
x0true = 0.5;
theta_true = 3.78;
Htrue = 1;
sigmaR = 1e-8; % for positive-definiteness
Qtrue = sigmaR^2; %1e-10 .5 .7 1.0 %changeable for experiment
Rtrue = sigmaR^2;
T = 200+1;
ytrue = zeros(1,T);
ytrue(:,1) = x0true;
for i = 1:T-1
    ytrue(:,i+1) = theta_true*ytrue(:,i)*(1-ytrue(:,i));
end
y = ytrue + sigmaR*randn(size(ytrue));

%% Specify learning problem
n = 1; % state dimension
m = 1; % measurement dimension
ptot = 1; % number of learnable parameters

indx0 = []; indP0 = []; indPsi = 1; indH = [];
indQ = []; indR = [];

x0 = readStructure(x0true,false(n,1),indx0);
P0 = readStructure(1e-16*eye(n), false(n),indP0);
Psi = @(theta,x)dynamicsModel(theta,x,indPsi);
H = readStructure(Htrue,false(m,n),indH);
Q = readStructure(Qtrue,false(n),indQ);
R = readStructure(Rtrue,false(n),indR);
lp = @(theta)struct('val',0); % no prior

%% Create objectives for Bayes and MS
% UKF parameters
alpha = 1e-3;
beta = 2;
kappa = 0;
lambda = alpha^2 * (n+kappa) - n;
[Wm, Wc] = formWeights(n, lambda, alpha, beta);
evalNLP = @(theta)lpNLDyn(x0(theta).val, P0(theta).val, ...
    @(x)Psi(theta,x).val, H(theta).val,[], ...
    Q(theta).val, R(theta).val, [], y, lp(theta).val, lambda,Wm,Wc); % Bayes

% changeable parameter for MS experiment; T in the paper
Tmax = 2; %T, 10, 5, 2
ms = @(theta)msObj(theta,y,Tmax,T); % MS

%% Run experiment
obj = @(theta)evalNLP(theta);
x = 2:0.001:4;
FVAL = zeros(size(x));
MS = zeros(size(x));
for i = 1:length(x)
    FVAL(i) = obj(x(i));
    MS(i) = ms(x(i));
end
%% Plot results
close all;
fontSize = 30; lineWidth = 1.5;
plotSettings(fontSize, lineWidth);

plot(x,FVAL*1/FVAL(1))
hold on
plot(x,MS*1/MS(1));
truth = plot([3.78,3.78],ylim,'r--','LineWidth',1.5);
xlabel('$\theta$')
ylabel('$J(\theta)$')
obj(theta_true)
grid on;
ylim([0 min(3,max(ylim))])
xticks(2:0.5:4)

%% Helper functions
function ms = msObj(theta,y,Tmax,T)
    idx = 1:Tmax:T;
    yhat = y;
    for i = 1:Tmax
        yhat(idx+1) = theta*yhat(idx).*(1-yhat(idx));
        idx = idx + 1;
    end
    yhat = yhat(1:T);
    ms = mean((yhat-y).^2);
end

function f = dynamicsModel(theta,x,ind)
    f.val = theta(ind).*x.*(1-x);
end