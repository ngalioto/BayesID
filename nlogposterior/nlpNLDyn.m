% evaluates negative log-posterior and desired gradients
% nonlinear dynamics model; linear observation model
function [fval, grad] = nlpNLDyn(idx, m, P, f, H, D, Q, R, u, y, nlp,lambda,Wm,Wc)
    p = length(idx); 
    % check the log-prior is real-valued number
    if (isnan(nlp.val) || ~isreal(nlp.val))
        fval = Inf;
        grad = NaN*ones(p,1);
        return;
    elseif (isinf(nlp.val))
        fval = nlp.val;
        grad = NaN*ones(p,1);
        return;
    end

    % get dimensions
    pderror = 'Error: Covariance is no longer positive definite. Evaluation ended early.\n';
    [dy,T] = size(y); dx = length(m.val); Idy = eye(dy);
    
    % remove unwanted gradients
    x_grad = 1:dx; y_grad = 1:dy;
    x_grad = kron(dx*(idx-1), ones(1,dx)) + kron(ones(1,p),x_grad);
    y_grad = kron(dy*(idx-1), ones(1,dy)) + kron(ones(1,p),y_grad);
    m.grad = m.grad(x_grad); P.grad = P.grad(x_grad,:); 
    H.grad = H.grad(y_grad,:); nlp.grad = nlp.grad(idx);
    Q.grad = Q.grad(x_grad,:); R.grad = R.grad(y_grad,:);
    dynInput = (nargin(f) == 3); includeD = ~isempty(D);
    if (dynInput)
        f = @(x,u)f(x_grad,x,u);
    else
        f = @(x)f(x_grad,x);
    end
    if (includeD)
        D.grad = D.grad(y_grad,:);
    end
    
    % indices for reshaping during matrix multiplication
    [xbwd,xfwd,ybwd,yfwd] = getIndices(p,dx,dy);

    for i = 1:T
        if (includeD)
            v = propV(dy,dx,p,m,H,D,u(:,i));
        else
            v = propV(dy,dx,p,m,H);
        end

        v.val = y(:,i)-v.val; v.grad = -v.grad;
        S = propM(dy,dx,p,P,H,R,ybwd,yfwd);
        Sinv = getSinv(dy,p,Idy,S,ybwd,yfwd);
        
        nlp = propLogpost(dy,p,nlp,S,Sinv,v,ybwd,yfwd);

        if (~isreal(nlp.val) || isnan(nlp.val))
            fval = Inf;
            grad = NaN*ones(p,1);
            return;
        elseif (isinf(nlp.val))
            fval = nlp.val;
            grad = NaN*ones(p,1);
            return;
        end
        
        if (i < T)
            K = getGain(dy,dx,p,P,H,Sinv,ybwd,yfwd);
            [m,P] = update_grad(dy,dx,p,m,P,K,H,v,xbwd,xfwd);
            
            if (dynInput)
                [m,err,resx] = propV_UKF(dx,p,m,P,f,u(:,i),Wm,lambda,xbwd,xfwd);
            else
                [m,err,resx] = propV_UKF(dx,p,m,P,f,[],Wm,lambda,xbwd,xfwd);
            end
            if (err ~= 0)
%                 fprintf(2,pderror);
                fval = Inf; grad = NaN*ones(p,1);
                return;
            end
            P = propM_UKF(dx,dx,p,resx,resx,Q,Wc,xfwd);
        end
    end
    fval = nlp.val;
    grad = nlp.grad;
end