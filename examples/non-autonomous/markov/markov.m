addpath(genpath('../../nlogposterior'))
addpath('../../utils')
clear;
m = 2;
n = 5;
p = 3;
nbar = 18;
T1 = 9; T2 = 8;
K = 1000; % number of subtrajectories
N = nbar+K-1; % number of data used for training
ind = (m*(nbar-1)+1):m*N;

A = eye(n);
B = randn(n,p)/sqrt(n);
C = randn(m,n)/sqrt(m);
D = randn(m,p)/sqrt(m);

% parameter indices of each learnable matrix/vector
indx0 = 1:n;
indA = (1:n^2)+indx0(end); indB = (1:n*p)+indA(end); indC = (1:m*n)+indB(end);
indD = (1:m*p)+indC(end); indQ = 1+indD(end); indR = 1+indQ(end);
pdyn = indD(end); ptot = indR(end);

% create estimation problem for Bayesian algorithm
x0est = readStructure(zeros(n,1),true(n,1),indx0,ptot);
P0est = readStructure(1e-16*eye(n), false(n), [], ptot);
Aest = readStructure(zeros(n,n),true(n,n),indA,ptot);
Best = readStructure(zeros(n,p),true(n,p),indB,ptot);
Cest = readStructure(zeros(m,n),true(m,n),indC,ptot);
Dest = readStructure(zeros(m,p),true(m,p),indD,ptot);
Q = readStructure(zeros(n,n),diag(true(n,1)),indQ,ptot);
R = readStructure(zeros(m,m),diag(true(m,1)),indR,ptot);
nlp = @(theta)formPrior(theta,indQ,indR,ptot);
npost = @(th,u,y)nlpLinear(1:ptot, x0est(th), P0est(th), Aest(th), ...
    Best(th), Cest(th), Dest(th), Q(th), R(th), u, y, nlp(th));
theta_init = [randn(pdyn,1); abs(randn(ptot-pdyn,1))];

Ns = 1;  % number of datasets to use (50 in paper)
numG = p*nbar;
Karr = numG:K;
Karr_map = [numG:100:K, K];
GLSerr = zeros(Ns,length(Karr));
LSerr = zeros(Ns,length(Karr));
MAPerr = zeros(Ns,length(Karr_map));
numK = length(Karr);
numK_map = length(Karr_map);
sigw = 0.25; sigz = 0.25;
% W is inverse Lambda in the paper
[W,sqrtW] = getWeights(A,C,sigw^2*eye(n),sigz^2*eye(m),N);
W = sparse(W); sqrtW = sparse(sqrtW);
for i = 1:Ns
    %% create data
    warning off;
    u = randn(p,N); % generate inputs
    [y,G] = collectOutput(A,B,C,D,u,sigw,sigz); % get Markov params and outputs
    G = G(:,1:p*nbar); % use only first nbar Markov params
    U = formU(u,K,nbar);
    V = kron(U,eye(m))';
    Y = formY(y,K,nbar);

    theta0 = theta_init;
    for j = 1:numK
        k = Karr(j);
        ind = (m*(nbar-1)+1):m*(nbar+k-1);
        G_ls = Y(:,1:k)/U(:,1:k);
        G_gls = pinv(V(1:m*k,:)'*W(ind,ind)*V(1:m*k,:))*V(1:m*k,:)'*W(ind,ind)*reshape(Y(:,1:k),[m*k,1]);
        G_gls = reshape(G_gls,[p*nbar,m])';
        GLSerr(i,j) = norm(G_gls-G);
        LSerr(i,j) = norm(G_ls-G);

        % check if this k-value was selected to train the bayesian approach
        Karr_bool = (k == Karr_map);
        if any(Karr_bool)
            j_map = find(Karr_bool);
            t_ind = nbar:(nbar+k-1);
            obj = @(theta)npost(theta,u(:,t_ind),y(:,t_ind));
            opt = optimoptions('fmincon','Display','final','SpecifyObjectiveGradient',true);
            success = false;
            while (~success)
                try
                    theta0 = fmincon(obj,theta0,[],[],[],[],[-Inf*ones(pdyn,1);zeros(2,1)],[],[],opt);
                    [~,G_map] = collectOutput(Aest(theta0).val,Best(theta0).val,Cest(theta0).val,Dest(theta0).val,u(:,1:nbar),0,0);
                    MAPerr(i,j_map) = norm(G_map-G);  
                    if (MAPerr(i,j_map) < LSerr(i,j))
                        success = true;
                    else
                        fprintf('Stopped at poor minimum. Trying new point\n');
                        theta0 = [randn(pdyn,1); abs(randn(ptot-pdyn,1))];
                    end
                catch
                    fprintf('Ran into error. Trying new point\n');
                    theta0 = [randn(pdyn,1); abs(randn(ptot-pdyn,1))];
                end
            end
            fprintf('Finished iteration %d/%d, sample number %.d/%d\n',j,numK,i,Ns);
        end
    end
    fprintf('Finished sample %d/%d\n',i,Ns);
end

function [y,G] = collectOutput(A,B,C,D,u,sigw,sigz)
    T = size(u,2);
    [m,n] = size(C);
    p = size(B,2);
    w = sigw*randn(n,T);
    z = sigz*randn(m,T);
    y = zeros(m,T);
    x = zeros(n,1);
    G = zeros(n,p*T);
    idx = 1:p;
    G(:,idx+p) = B;
    y(:,1) = D*u(:,1) + z(:,1);
    for i = 2:T
        idx = idx + p;
        x = A*x + B*u(:,i-1) + w(:,i-1);
        y(:,i) = C*x + D*u(:,i) + z(:,i);
        if (i > 2)
            G(:,idx) = A*G(:,idx-p);
        end
    end
    G = C*G;
    G(:,1:p) = D;
end

function [W,sqrtW] = getWeights(A,C,Sigma,Gamma,T)
    [m,~] = size(C);
    W = zeros(m,m,T);
    sqrtW = zeros(m,m,T);
    W(:,:,1) = inv(Gamma);
    sqrtW(:,:,1) = chol(W(:,:,1));
    V = Sigma;
    sumV = V;
    W(:,:,2) = inv(C*sumV*C'+Gamma);
    sqrtW(:,:,2) = chol(W(:,:,2));
    for i = 3:T
        V = A*V*A';
        sumV = sumV + V;
        W(:,:,i) = inv(C*sumV*C'+Gamma);
        sqrtW(:,:,i) = chol(W(:,:,i));
    end
    q = num2cell(W,[1,2]);
    W = blkdiag(q{:});
    q = num2cell(sqrtW,[1,2]);
    sqrtW = blkdiag(q{:});
end

% maybe put into utils folder
function nlp = formPrior(theta,indQ,indR,ptot)
    Q = theta(indQ); R = theta(indR);
    if (sum(Q<0) > 0 || sum(R<0) > 0)
        nlp.val = Inf;
        nlp.grad = nan*ones(ptot,1);
    else
        nlp.grad = zeros(ptot,1);
        nlp.val = 0.5*(Q'*Q + R'*R);
        nlp.grad(indQ) = Q;
        nlp.grad(indR) = R;
    end
end