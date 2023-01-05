% evaluates positive log-posterior
% linear dynamics model; linear observation model
function lp = lpNonlinear(m, P, f, h, Q, R, u, y, lp, lambda, Wm, Wc)
    % check the log-prior is real-valued number
    if (isnan(lp) || ~isreal(lp))
        lp = -Inf;
        return;
    elseif (isinf(lp))
        return;
    end

    % get dimensions and model form
    pderror = 'Error: Covariance is no longer positive definite. Evaluation ended early.\n';
    [dy,T] = size(y); dx = length(m); Idy = eye(dy);
    dynInput = (nargin(f) == 2); obsInput = (nargin(h) == 2);

    for i = 1:T
        if (obsInput)
            [lp,PH,Sinv,Sinvv] = getLogpost_UKF(lp,dx,m,P,R,h,u(:,i),y(:,i),Idy,Wm,Wc,lambda);
        else
            [lp,PH,Sinv,Sinvv] = getLogpost_UKF(lp,dx,m,P,R,h,[],y(:,i),Idy,Wm,Wc,lambda);
        end
        
        if (~isreal(lp) || isnan(lp))
            lp = -Inf;
            return;
        elseif (isinf(lp))
            return;
        end
        
        if (i < T)
            [m,P] = getUpdate(m,P,PH',Sinv,Sinvv);
            % predict
            if (dynInput)
                [m,P,err] = predictUKF(dx,m,P,f,Q,u(:,i),Wm,Wc,lambda);
            else
                [m,P,err] = predictUKF(dx,m,P,f,Q,[],Wm,Wc,lambda);
            end
            if (err ~= 0)
                fprintf(2,pderror);
                lp = -Inf;
                return;
            end
        end
    end
end