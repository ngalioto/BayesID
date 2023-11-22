function K = getGain(m,n,ptot,U,Sinv,indbwd,indfwd)
    gradReshape = reshape(Sinv.grad,[m,m*ptot]);
    gradReshape = U.val*gradReshape(:,indfwd);
    term1 = reshape(gradReshape(:,indbwd),[n*ptot,m]);
    
    K.grad = U.grad*Sinv.val + term1;
    K.val = U.val*Sinv.val;
end