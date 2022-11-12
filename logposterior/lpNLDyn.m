% evaluates positive log-posterior
% nonlinear dynamics model; linear observation model
function lp = lpNLDyn(m, P, f, H, D, Q, R, u, y, lp, Wm, Wc, lambda)
%     addpath('C:/Users/Nick/Documents/MATLAB/nickfiltering/BayesianID/filtering')
    % check the log-prior is real-valued number
    if (isnan(lp) || ~isreal(lp))
        lp = -Inf;
        return;
    elseif (isinf(lp))
        return;
    end

    % get dimensions
    pderror = 'Error: Covariance is no longer positive definite. Evaluation ended early.\n';
    [dy,T] = size(y); dx = length(m); Idy = eye(dy);

    dynInput = (nargin(f) == 2); includeD = ~isempty(D);

    for i = 1:T
        if (includeD)
            [lp,HP,Sinv,Sinvv] = getLogpost(lp,m,P,H,R,y(:,i),Idy,D,u(:,i));
        else
            [lp,HP,Sinv,Sinvv] = getLogpost(lp,m,P,H,R,y(:,i),Idy);
        end

        if (~isreal(lp) || isnan(lp))
            lp = -Inf;
            return;
        elseif (isinf(lp))
            return;
        end
        
        if (i < T)
            % update
            [m,P] = getUpdate(m,P,HP,Sinv,Sinvv);
            
            if (dynInput)
                [m,P,err] = predictUKF(dx,m,P,f,Q,u(:,i),Wm,Wc,lambda);
            else
                [m,P,err] = predictUKF(dx,m,P,f,Q,[],Wm,Wc,lambda);
            end
            if (err ~= 0)
%                 fprintf(2,pderror);
                lp = -Inf;
                return;
            end
        end
    end
end