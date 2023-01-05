% evaluates positive log-posterior
% linear dynamics model; linear observation model
function lp = lpNLObs(m, P, A, B, h, Q, R, u, y, lp, lambda, Wm, Wc)
    addpath('../filtering')
    % check the log-prior is real-valued number
    if (isnan(lp) || ~isreal(lp))
        lp = -Inf;
        return;
    elseif (isinf(lp))
        return;
    end

    % get dimensions and model form
    [dy,T] = size(y); dx = length(m); Idy = eye(dy);
    includeB = ~isempty(B); obsInput = (nargin(h) == 2);

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
            [m,P] = getUpdate(m,P,PH,Sinv,Sinvv);
            % predict
            if (includeB)
                [m,P] = predictKF(m,P,A,Q,B,u(:,i));
            else
                [m,P] = predictKF(m,P,A,Q);
            end
        end
    end
end