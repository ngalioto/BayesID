function K = getGain(m,n,ptot,C,H,Sinv,indbwd,indfwd)
    CH = C.val*H.val';
    
    gradReshape = C.val*H.grad';
    term1 = reshape(gradReshape(:,indbwd),[n*ptot,m])*Sinv.val;
    gradReshape = reshape(Sinv.grad,[m,m*ptot]);
    gradReshape = CH*gradReshape(:,indfwd);
    term2 = reshape(gradReshape(:,indbwd),[n*ptot,m]);
    K.grad = C.grad*H.val'*Sinv.val + term1 + term2;
    
    K.val = CH*Sinv.val;
end