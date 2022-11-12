function mat = propM_UKF(d1,d2,ptot,res1,res2,nvar,Wc,indfwd)
    opgrad2 = Wc.*res1.val*res2.grad';
    mat.grad = (Wc.*res1.grad*res2.val' + reshape(opgrad2(:,indfwd),[d1*ptot,d2]));
    mat.val = Wc .* res1.val*res2.val';
    if (~isempty(nvar))
        mat.grad = mat.grad + nvar.grad;
        mat.val = mat.val + nvar.val;
    end
end