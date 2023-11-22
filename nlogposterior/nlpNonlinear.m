% evaluates negative log-posterior and desired gradients
% nonlinear dynamics model; nonlinear observation model
function [fval, grad] = nlpNonlinear(idx, m, P, f, h, Q, R, u, y, nlp,lambda,Wm,Wc)
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
    
    pderror = 'Error: Covariance is no longer positive definite. Evaluation ended early.\n';
    [dy,T] = size(y); dx = length(m.val);
    [xbwd,xfwd,ybwd,yfwd] = getIndices(p,dx,dy);
    
    x_grad = 1:dx; y_grad = 1:dy;
    x_grad = kron(dx*(idx-1), ones(1,dx)) + kron(ones(1,p),x_grad);
    y_grad = kron(dy*(idx-1), ones(1,dy)) + kron(ones(1,p),y_grad);
    m.grad = m.grad(x_grad); P.grad = P.grad(x_grad,:); nlp.grad = nlp.grad(idx);
    Q.grad = Q.grad(x_grad,:); R.grad = R.grad(y_grad,:);
    dynInput = (nargin(f) == 3); obsInput = (nargin(h) == 3);
    if (dynInput)
        f = @(x,u)f(idx,x,u);
    else
        f = @(x)f(idx,x);
    end
    if (obsInput)
        h = @(x,u)h(idx,x,u);
    else
        h = @(x)h(idx,x);
    end
    
    Id = eye(dy);
    for i = 1:T
        if (obsInput)
            [v,err,resy,resx] = propV_UKF(dx,p,m,P,h,u(:,i),Wm,lambda,xbwd,xfwd);
        else
            [v,err,resy,resx] = propV_UKF(dx,p,m,P,h,[],Wm,lambda,xbwd,xfwd);
        end

        if (err ~= 0)
            fprintf(2,pderror);
            fval = Inf;
            grad = NaN*ones(p,1);
            return;
        end
        
        v.val = y(:,i)-v.val; v.grad = -v.grad;
        S = propM_UKF(dy,dy,p,resy,resy,R,Wc,yfwd);
        
        Sinv = getSinv(dy,p,Id,S,ybwd,yfwd);
        
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
            U = propM_UKF(dx,dy,p,resx,resy,[],Wc,yfwd);
            K = getGain(dy,dx,p,U,Sinv,ybwd,yfwd);
            [m,P] = update_grad(dy,dx,p,m,P,U,K,v,xbwd,xfwd);
        
            if (dynInput)
                [m,err,resx] = propV_UKF(dx,p,m,P,f,u(:,i),Wm,lambda,xbwd,xfwd);
            else
                [m,err,resx] = propV_UKF(dx,p,m,P,f,[],Wm,lambda,xbwd,xfwd);
            end
            if (err ~= 0)
                fprintf(2,pderror);
                fval = Inf; grad = NaN*ones(p,1);
                return;
            end
            P = propM_UKF(dx,dx,p,resx,resx,Q,Wc,xfwd);
        end
    end
    fval = nlp.val;
    grad = nlp.grad;
end