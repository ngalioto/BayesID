function Sinv = getSinv(m,ptot,Id,S,indbwd,indfwd)
    Sinv.val = Id / S.val;
    gradReshape = reshape(S.grad*Sinv.val,[m,m*ptot]);
    Sinv.grad = -Sinv.val * gradReshape(:,indfwd);
    Sinv.grad = reshape(Sinv.grad(:,indbwd),[m*ptot,m]);
end