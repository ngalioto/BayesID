function [grad,CO] = gradM(m,n,ptot,matCov, matOp, matVar,indbwd,indfwd)
    CO = matCov.val*matOp.val';
    term0 = matOp.grad*CO;

    gradReshape = reshape(matCov.grad*matOp.val',[n,m*ptot]);
    gradReshape = matOp.val * gradReshape(:,indfwd);
    term1 = reshape(gradReshape(:,indbwd),[m*ptot,m]);
    
    gradReshape = reshape(term0',[m,m*ptot]);
    term2 = reshape(gradReshape(:,indbwd),[m*ptot,m]);
    grad = term0 + term1 + term2;
    if (~isempty(matVar))
        grad = grad + matVar.grad;
    end
end