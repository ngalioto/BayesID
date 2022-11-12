% evaluates negative log-posterior and desired gradients
% linear dynamics model; nonlinear observation model
function [fval, grad] = nlpNLObs(idx, m, P, A, B, h, Q, R, u, y, nlp,lambda,Wm,Wc)
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
    [dy,T] = size(y); dx = length(m.vec);
    [xbwd,xfwd,ybwd,yfwd] = getIndices(p,dx,dy);
    
    x_grad = 1:dx; y_grad = 1:dy;
    x_grad = kron(dx*(idx-1), ones(1,dx)) + kron(ones(1,p),x_grad);
    y_grad = kron(dy*(idx-1), ones(1,dy)) + kron(ones(1,p),y_grad);
    m.grad = m.grad(x_grad); P.grad = P.grad(x_grad,:); A.grad = A.grad(x_grad,:);
    Q.grad = Q.grad(x_grad,:); R.grad = R.grad(y_grad,:); nlp.grad = nlp.grad(idx);
    includeB = ~isempty(B); obsInput = (nargin(h) == 3);
    if (includeB)
        B.grad = B.grad(x_grad,:);
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
        v.vec = y(:,i)-v.vec; v.grad = -v.grad;
        if (err ~= 0)
            fprintf(2,pderror);
            fval = Inf;
            grad = NaN*ones(p,1);
            return;
        end
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
            C = propM_UKF(dx,dy,p,resx,resy,[],Wc,yfwd);
            K = getGain(dy,dx,p,C,Sinv,ybwd,yfwd);
            [m,P] = update_grad(dy,dx,p,m,P,S,K,v,xbwd,xfwd);
        
            if (includeB)
                m = propV(dx,dx,p,m,A,B,u(:,i));
            else
                m = propV(dx,dx,p,m,A);
            end
            P = propM(dx,dx,p,P,A,Q,xbwd,xfwd);
        end
    end
    fval = nlp.val;
    grad = nlp.grad;
end