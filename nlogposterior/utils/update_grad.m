function [x,C] = update_grad(m,n,ptot,x,C,K,H,v,indbwd,indfwd)
    HC = H.val*C.val;

    gradReshape = -K.val*reshape(v.grad,[m,ptot]);
    x.grad = x.grad + K.grad*v.val - reshape(gradReshape,[n*ptot,1]);
    
    gradReshape = reshape(H.grad*C.val,[m,n*ptot]);
    gradReshape = K.val * gradReshape(:,indfwd);
    term1 = reshape(gradReshape(:,indbwd),[n*ptot,n]);
    gradReshape = reshape(C.grad,[n,n*ptot]);
    gradReshape = K.val*H.val*gradReshape(:,indfwd);
    term2 = reshape(gradReshape(:,indbwd),[n*ptot,n]);
    C.grad = C.grad - K.grad*HC - term1 - term2;
    
    x.val = x.val + K.val*v.val;
	C.val = C.val - K.val*HC;
end