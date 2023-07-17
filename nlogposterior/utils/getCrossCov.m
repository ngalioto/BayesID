function U = getCrossCov(m,n,ptot,P,H,indbwd)
    gradReshape = P.val*H.grad';
    term0 = reshape(gradReshape(:,indbwd),[n*ptot,m]);
    
    U.grad = P.grad*H.val' + term0;
    U.val = P.val*H.val';
end