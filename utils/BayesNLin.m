function [samples, accRatio, propC] = BayesNLin(y, theta, propC, M, m0, Sigma0, f, h, Q, R, logprior, n0, gamma, eps)
    % [samples, accRatio] = BayesNLin(y, theta, propC, M, m0, Sigma0, f, h, Q, R, logprior, n0, gamma, eps);
    % [samples, accRatio] = BayesNLin(y, theta, propC, M, m0, Sigma0, f, h, Q, R, logprior);
    % Nick Galioto. University of Michigan.  Dec. 20, 2019
    %
    % Let n be state dimension, m be measurement dimension, p be the
    %     parameter dimension, and N the number of measurements
    % Inputs:
    %
    %       y           Matrix of data. Size [m x N].
    %
    %       theta       Starting point of the MCMC sampler. Size [p x 1].
    %
    %       propC       Starting proposal covariance of MCMC sampler. Size
    %                   [p x p].
    %
    %       M           Number of samples to be drawn from the posterior
    %
    %       m0          Vector of the mean of the initial state.  Size [n x 1].
    %
    %       Sigma0      Covariance of the initial state. Size [n x n].
    %
    %       f           Discrete dynamics function. Accepts an [n x 1]
    %                   state vector and a [p x 1] parameter vector as input and
    %                   returns an [n x 1] state vector as output.  This
    %                   function should map the state from one measurement
    %                   time to the next.
    %
    %       h           Measurement function. Accepts an [n x 1] state 
    %                   vector and a [p x 1] parameter vector as input and
    %                   returns an [m x 1] observation vector as output.
    %
    %       Q           Process noise covariance matrix. Size [n x n]. Must
    %                   be a function of theta.
    %
    %       R           Measurement noise covariance matrix. Size [m x m].
    %                   Must be a function of theta.
    %
    %       logprior    Function of theta that returns the log of the 
    %                   posterior evaluated at theta.
    %
    %       n0          Number of samples to draw before starting the
    %                   adaptive covariance algorithm.  Default value of
    %                   200.
    %
    %       gamma       The scaling factor for the second level proposal
    %                   covariance.  Default value of 0.01.
    %
    %       eps          A small positive value to ensure covariance
    %                    matrices remain positive definite.  Default value
    %                    of 1e-10.
    %
    % Outputs:
    %
    %       samples     Matrix holding the samples drawn from the posterior. 
    %                   Size [p x M].
    %
    %       accRatio    Acceptance ratio.  Percentage of samples that get
    %                   accepted.
    %
    %       propC       Proposal covariance at the end of sampling.
    
    
    alpha = 1e-3;
    beta = 1;
    kappa = 0;
    p = length(theta);

    % default parameters for DRAM
    if (nargin < 14 || isempty(eps)) 
        eps = 1e-10;
    end
    if (nargin < 13 || isempty(gamma))
        gamma = 0.01;
    end
    if (nargin < 12 || isempty(n0))
        n0 = 200;
    end
    
    % Initialize variables for MCMC
    numacc = 0;
    samples = zeros(p, M);
    samples(:,1) = theta;
    theta_mean = theta;
    theta_sd = 2.4^2 / p;
    propC = theta_sd*propC;  % optimal assuming propC approximates posterior covariance
    logpost_eval = @(theta)PMlogpost(m0, Sigma0, @(x)f(x,theta), ...
            @(x)h(x,theta), Q(theta), R(theta), y, logprior(theta), alpha, beta, kappa,eps);
    logpost = logpost_eval(theta);
    
    % Begin sampling
    fprintf(1,'Computation Progress:%3.0f%%\n',100/M);
    for i = 2:M  
        [theta_mean, propC] = AdaptiveCov(theta, theta_mean, propC, p, i-1, n0, theta_sd, eps);
        if (det(propC) <= 0)
            propC = condCov(propC); 
        end
        [theta, acc,logpost] = DelayedRej(theta, propC, logpost_eval, gamma, logpost);
        samples(:,i) = theta;
        numacc = numacc + acc;
        fprintf(1,'\b\b\b\b%3.0f%%',100*i/M);% Deleting 4 characters (The three digits and the % symbol)
    end
    accRatio = numacc / M;
    fprintf('\n');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Computes the approximated log posterior using the pseudo-marginal 
% algorithm.  Approximation achieved using an unscented Kalman filter.  If
% the covariance matrix becomes not positive definite within the filter,
% the function halts and returns -Inf for the log posterior
function logpost = PMlogpost(m, C, f, h, Sigma, Gamma, y, logprior,alpha,beta,kappa,eps)
    T = size(y,2);
    n = size(m,1);
    logpost = logprior;
    if (isfinite(logpost))  % can immediately reject if logprior is -Inf
        lambda = alpha^2 * (n+kappa) - n;
        [Wm, Wc] = form_weights(n, lambda, alpha, beta);

        for i = 1:T
            [m, C,err] = UKFpredict(m, C, Sigma, n, f, lambda, Wm, Wc, eps);
            if (err ~= 0)
                logpost = -Inf;
                break;
            end
            [m,C,v,S,Sinv,err] = UKFupdate(m, C, y(:,i), Gamma, n, h, lambda, Wm, Wc, eps);
            if (err ~= 0)
                logpost = -Inf;
                break;
            end
            logpost = logpost - 0.5*log(det(S)) - 0.5*v'*Sinv*v;
        end
    end
end

% Performs the prediction step of the unscented Kalman filter.  If the
% covariance matrix is not positive definite, the function halts and
% returns a non-zero value in the 'err' variable
function [xout, Pout,err] = UKFpredict(xin, Pin, Q, n, f, lambda, Wm, Wc, eps)
    [X,err] = form_sigma_points(xin, Pin, n, lambda);
    if (err == 0)
    Xhat = zeros(size(X));
    for i = 1:2*n+1
        Xhat(:,i) = f(X(:,i));
    end
    
    xout = sum(Xhat .* Wm,2);
    Pout = Wc .* (Xhat - xout)*(Xhat - xout)' + Q + eps*eye(n);
    else
        xout = xin;
        Pout = Pin;
    end
end

% Performs the update step of the unscented Kalman filter.  If the
% covariance matrix is not positive definite, the function halts and
% returns a 1 in the 'err' variable
function [xout, Pout,v,S,Sinv,err] = UKFupdate(xin, Pin, y, R, n, h, lambda,Wm,Wc,eps)
    m = size(y, 1);
    [X,err] = form_sigma_points(xin, Pin, n, lambda);
    if (err == 0)
        Yhat = zeros(m,size(X,2));
        for i = 1:2*n+1
            Yhat(:,i) = h(X(:,i));
        end

        mu = sum(Wm .* Yhat,2);
        S = Wc .* (Yhat - mu)*(Yhat - mu)' + R + eps*eye(m);
        C = Wc .* (X - xin)*(Yhat - mu)';

        v = y - mu;
        Sinv = eye(m) / S;
        K = C*Sinv;
        xout = xin + K*v;
        Pout = Pin - K*S*K' + eps*eye(n);
    else
        xout = xin;
        Pout = Pin;
        v = 0;
        S = 0;
        Sinv = 0;
    end
end

% Forms the weights used in the unscented transform.
function [Wm, Wc] = form_weights(n, lambda, alpha, beta)
    % Form weights for unscented Kalman filter
    Wm = zeros(1, 2*n+1);
    Wc = zeros(1, 2*n+1);

    Wm(1) = lambda / (n+lambda);
    Wm(2:end) = 1 / (2*(n + lambda));
    Wc(1) = lambda / (n+lambda) + 1 - alpha^2 + beta;
    Wc(2:end) = 1 / (2*(n+lambda));
end

% Forms the sigma points used in the unscented transform.
function [xout,err] = form_sigma_points(xin, Pin, n, lambda)
	[cholP,err] = chol(Pin);
    if (err == 0)
        sqrtP = sqrt(n+lambda)*cholP;
        xout = zeros(length(xin), 2*n+1);
        xout(:, 1) = xin;
        for i = 1:n
            xout(:,i+1) = xin + sqrtP(i,:)';
            xout(:,i+n+1) = xin - sqrtP(i,:)';
        end
    else
        xout = xin;
    end
end

% Delayed rejection algorithm.  See Haario 2006.
function [xout, acc, logpost] = DelayedRej(xin, propC, post_eval, gamma, fx)
    acc = 0;
    
    if (nargin == 4)
        fx = post_eval(xin);
    end
    logpost = fx;
    
	y1 = mvnrnd(xin, propC)';
    fy1 = post_eval(y1);
    
    alphay1_x = min(fy1 - fx, 0); % acceptance probability
    if (log(rand) < alphay1_x)  % acceptance
        xout = y1;
        acc = 1;
        logpost = fy1;
    else                        % 1st rejection
        y2 = mvnrnd(xin, gamma*propC)';
        fy2 = post_eval(y2);
        
        qx_y1 = log(mvnpdf(xin, y1, propC));
        q1y1_y2 = log(mvnpdf(y1, y2, propC));
        
        alphay1_y2 = min(fy1 - fy2, 0);
        N2 = fy2 + q1y1_y2 + log(1-exp(alphay1_y2));
        D2 = fx + qx_y1 + log(1-exp(alphay1_x));
        
        alpha2 = N2 - D2; % accetance probability
        if (log(rand) < alpha2) % acceptance
            xout = y2;
            acc = 1;
            logpost = fy2;
        else                    % 2nd rejection
            xout = xin;
        end
    end
end

% Adaptive covariance algorithm. See Haario 2006.
function [muout, Cout] = AdaptiveCov(x, muin, Cin, d, n, n0, sd, eps)
    muout = ((n-1)*muin + x) / n;
    if n > n0
        term1 = ((n-1)/n) * Cin;
        term2 = (n*(muin*muin') - (n+1)*(muout*muout'));
        term3 = (x*x') + eps*eye(d);

        Cout = term1 + (sd/n)*(term2 + term3); 
    else
        Cout = Cin;
    end
end

% Adds nuggets of increasing magnitude to a covariance matrix until the 
% matrix is positive definite.
function Pout = condCov(Pin)
    n = size(Pin,1);
    i = -8;
    while (sum(eig(Pin + 10^i*eye(n)) <= 0) > 0)
        i = i + 1;
    end
    Pout = Pin + 10^i*eye(n);
end
