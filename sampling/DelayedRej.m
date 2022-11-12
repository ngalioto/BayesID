% Delayed rejection algorithm.  See Haario 2006.
% Assumes Gaussian proposals
function [xout, acc, logpost] = DelayedRej(xin, sqrtC, post_eval, gamma, fx)
    acc = 0;
    n = size(xin,1);
    
    if (nargin == 4)
        fx = post_eval(xin);
    end
    logpost = fx;
    
    y1 = sqrtC*randn(n,1) + xin;
    fy1 = post_eval(y1);
    
    alphay1_x = min(fy1 - fx, 0); % acceptance probability
    if (log(rand) < alphay1_x)  % acceptance
        xout = y1;
        acc = 1;
        logpost = fy1;
    else                        % 1st rejection
        y2 = sqrt(gamma)*(sqrtC*randn(n,1)) + xin;
        fy2 = post_eval(y2);
        

        qx_y1 = sum((sqrtC\(xin-y1)).^2);
        q1y1_y2 = sum((sqrtC\(y1-y2)).^2);
        
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