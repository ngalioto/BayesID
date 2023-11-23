function [samples,accR,propC] = Gibbs_DRAM(idx,theta,propC,M,evalLogPost,n0,gamma,eps)

    % default parameters for DRAM
    if (nargin < 8 || isempty(eps)) 
        eps = 1e-10;
    end
    if (nargin < 7 || isempty(gamma))
        gamma = 0.01;
    end
    if (nargin < 6 || isempty(n0))
        n0 = 200;
    end
    
    N = length(idx);
    p = zeros(N,1);
    theta_mean = theta;
    theta_sd = zeros(N,1);
    sqrtC = cell(1,N);
    for j = 1:N    
        p(j) = length(idx{j});
        theta_sd(j) = 2.4^2 / p(j);
        propC{j} = theta_sd(j) * propC{j};
        sqrtC{j} = chol(propC{j},'lower');
    end
    
    samples = zeros(sum(p),M);
    samples(:,1) = theta;
    logpost = zeros(N,1);
    for j = 1:N
        logpost(j) = evalLogPost{j}(theta(idx{j}),theta);
    end
    numacc = zeros(N,1);
    printPercent = 1;
    next_percent = printPercent;
    progress = waitbar(0, '0', 'Name', 'Sampling the posterior...');
    for i = 2:M
        for j = 1:N
            condPost = @(x)evalLogPost{j}(x,theta);
            [theta(idx{j}), acc,logpost(j)] = DelayedRej(theta(idx{j}), sqrtC{j}, condPost, gamma, logpost(j));
            samples(idx{j},i) = theta(idx{j});
            numacc(j) = numacc(j) + acc;
            if (i >= n0)
                [theta_mean(idx{j}), propC{j}] = AdaptiveCov(samples(idx{j},:), theta_mean(idx{j}), propC{j}, p(j), i, n0, theta_sd(j), eps);
                [sqrtC{j}, pd] = chol(propC{j},'lower');
                if (pd ~= 0)
                    [propC{j},sqrtC{j}] = conditionCov(propC{j},eps);
                end
            end
        end
        percent_complete = 100*i/M;
        if (percent_complete >= next_percent)
            waitbar(percent_complete/100, progress, sprintf('%g',percent_complete));
            next_percent = floor(percent_complete) + printPercent;
        end
    end
    accR = numacc / (M-1);
    close(progress);
end