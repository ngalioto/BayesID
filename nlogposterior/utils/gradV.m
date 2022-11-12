function grad = gradV(m,n,ptot,vec,mat)   
    gradReshape = reshape(vec.grad,[n,ptot]);
    grad = mat.grad*vec.val + reshape(mat.val*gradReshape,[m*ptot,1]);
end