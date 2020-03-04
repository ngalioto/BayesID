function [samples, accRatio] = BayesLin(y, theta, propC, M, m0, Sigma0, A, H, Q, R, logprior, n0, gamma, eps)    
    % [samples, accRatio] = BayesLin(y, theta, propC, M, m0, Sigma0, A, H, Q, R, logprior, n0, gamma, eps);
    % [samples, accRatio] = BayesLin(y, theta, propC, M, m0, Sigma0, A, H, Q, R, logprior);
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
    %       M           Number of samples to be drawn from the posterior.
    %
    %       m0          Vector of the mean of the initial state.  Size [n x 1].
    %
    %       Sigma0      Covariance of the initial state. Size [n x n].
    %
    %       A           Linear discrete dynamics matrix. Size [n x n]. Must be 
    %                   a function of theta.
    %
    %       H           Linear measurement matrix. Size [m x n]. Must be a
    %                   function of theta.
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
    p = length(theta);
    numacc = 0;
    samples = zeros(p, M);
    samples(:,1) = theta;
    theta_mean = theta;
    theta_sd = 2.4^2 / p;
    propC = theta_sd*propC;
    logpost_eval = @(theta)PMlogpost(theta, m0, Sigma0, A(theta), H(theta), Q(theta), R(theta), y, logprior(theta));
    logpost = logpost_eval(theta);
    
    % Begin sampling
    for i = 2:M      
        [theta_mean, propC] = AdaptiveCov(theta, theta_mean, propC, p, i-1, n0, theta_sd, eps);
        if (det(propC) <= 0)
            propC = condCov(propC); 
        end
        [theta, acc,logpost] = DelayedRej(theta, propC, logpost_eval, gamma, logpost);
        samples(:,i) = theta;
        numacc = numacc + acc;
    end
    accRatio = numacc / M;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Computes the log posterior using the pseudo-marginal algorithm.
function logpost = PMlogpost(m, C, A, H, Sigma, Gamma, y, logprior)
    [d,T] = size(y);
    
    logpost = logprior;
    for i = 1:T
        m = A*m;
        C = A*C*A' + Sigma;
        
        mu = H*m;
        S = H*C*H' + Gamma;
        
        logpost = logpost - 0.5*log(det(S)) - 0.5*d*log(2*pi) -  ...
            0.5*((y(:,i)-mu)'/S*(y(:,i)-mu));
        
        K = C*H' / S;
        m = m + K *(y(:,i)-H*m);
        C = C - K*H*C;
    end
end

% Delayed rejection algorithm.  See Haario 2006.
function [xout, acc, logpost] = DelayedRej(xin, propC, post_eval, gamma, fx)
    acc = 0;
    y1 = mvnrnd(xin, propC)';

    if (nargin == 4)
        fx = post_eval(xin);
    end
    logpost = fx;
    fy1 = post_eval(y1);
    qx_y1 = log(mvnpdf(xin, y1, propC));
    qy1_x = log(mvnpdf(y1, xin, propC));
    
    N1 = fy1 + qx_y1;
    D1 = fx + qy1_x;
    alphay1_x = N1 - D1; % acceptance probability
    if (log(rand) < alphay1_x)  % acceptance
        xout = y1;
        acc = 1;
        logpost = fy1;
    else                        % 1st rejection
        y2 = mvnrnd(xin, gamma*propC)';
        
        fy2 = post_eval(y2);
        q1y1_y2 = log(mvnpdf(y1, y2, propC));
        q1y2_y1 = log(mvnpdf(y2, y1, propC));
        q2y2_x = log(mvnpdf(y2, xin, gamma*propC));
        q2x_y2 = log(mvnpdf(xin, y2, gamma*propC));
        
        alphay1_y2 = (fy1 + q1y2_y1) - (fy2 + q1y1_y2);
        N2 = fy2 + q1y1_y2 + q2x_y2 + log(1 - exp(alphay1_y2));
        
        alpha2 = N2 - (q2y2_x + log(exp(D1)-exp(N1))); % accetance probability
        if (log(rand) < alpha2) % acceptance
            xout = y2;
            acc = 1;
            logpost = fy2;
        else                    % 2nd rejection
            xout = xin;
        end
    end
end

% Adaptive covariance algorithm.  See Haario 2006.
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