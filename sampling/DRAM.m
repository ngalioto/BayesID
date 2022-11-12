function [samples,accR,propC] = DRAM(theta,propC,M,evalLogPost,n0,gamma,epsilon)

    % default parameters for DRAM
    if (nargin < 7 || isempty(epsilon)) 
        epsilon = 1e-10;
    end
    if (nargin < 6 || isempty(gamma))
        gamma = 0.01;
    end
    if (nargin < 5 || isempty(n0))
        n0 = 200;
    end

    p = length(theta);
    theta_mean = theta;
    theta_sd = 2.4^2 / p;
    propC = theta_sd * propC;
    sqrtC = chol(propC,'lower');
    
    samples = zeros(p,M);
    samples(:,1) = theta;
    logpost = evalLogPost(theta);
    numacc = 0;
    fprintf(1,'Computation Progress:%3.0f%%\n',100/M);
    printPercent = 1;
    next_percent = printPercent;
    for i = 2:M      
        [theta, acc,logpost] = DelayedRej(theta, sqrtC, evalLogPost, gamma, logpost);
        samples(:,i) = theta;
        numacc = numacc + acc;
        if (i >= n0)
            [theta_mean, propC] = AdaptiveCov(samples, theta_mean, propC, p, i, n0, theta_sd, epsilon);
            [sqrtC,pd] = chol(propC);
            if (pd ~= 0)
                [propC,sqrtC] = conditionCov(propC,epsilon); 
            end
        end
        percent_complete = 100*i/M;
        if (percent_complete >= next_percent)
%             fprintf(1,'\b\b\b\b%3.0f%%',percent_complete);% Deleting 4 characters (The three digits and the % symbol)
            fprintf(1,'Computation Progress:%3.0f%%\n',percent_complete);
            next_percent = floor(percent_complete) + printPercent;
        end
    end
    accR = numacc / (M-1);
    fprintf('\n');
end