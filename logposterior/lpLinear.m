% evaluates positive log-posterior
% linear dynamics model; linear observation model
function lp = lpLinear(m, P, A, B, H, D, Q, R, u, y, lp)
    % check the log-prior is real-valued number
    if (isnan(lp) || ~isreal(lp))
        lp = -Inf;
        return;
    elseif (isinf(lp))
        return;
    end

    % get dimensions and model form
    [dy,T] = size(y); Idy = eye(dy);
    includeB = ~isempty(B); includeD = ~isempty(D);

    for i = 1:T
        if (includeD)
            [lp,PH,Sinv,Sinvv] = getLogpost(lp,m,P,H,R,y(:,i),Idy,D,u(:,i));
        else
            [lp,PH,Sinv,Sinvv] = getLogpost(lp,m,P,H,R,y(:,i),Idy);
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