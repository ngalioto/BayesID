function [x,C] = update_grad(m,n,ptot,x,C,U,K,v,indbwd,indfwd)
    gradReshape = -K.val*reshape(v.grad,[m,ptot]);
    x.grad = x.grad + K.grad*v.val - reshape(gradReshape,[n*ptot,1]);
    
    gradReshape = reshape(U.grad,[m,n*ptot]);
    gradReshape = K.val*gradReshape(:,indfwd);
    term1 = reshape(gradReshape(:,indbwd),[n*ptot,n]);
    C.grad = C.grad - K.grad*U.val' - term1;
    
    x.val = x.val + K.val*v.val;
	C.val = C.val - K.val*U.val';
end