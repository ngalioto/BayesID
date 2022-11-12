function mat = propM(m,n,p,cov,op,nvar,indbwd,indfwd)
    [mat.grad,CO] = gradM(m,n,p,cov,op,nvar,indbwd,indfwd);
    mat.val = op.val*CO;
    if (~isempty(nvar))
        mat.val = mat.val + nvar.val;
    end
end