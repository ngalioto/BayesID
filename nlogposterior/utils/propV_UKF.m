function [vout,err,res1,res2] = propV_UKF(n,p,vec,cov,func,u,Wm,lambda,indbwd,indfwd)
    [sigmaPts,err] = formSigmaPoints_grad(vec, cov, n, p, lambda,indbwd,indfwd);
    if (err == 0)
        if (nargout == 4)
            res2.grad = sigmaPts.grad - vec.grad;
            res2.val = sigmaPts.val - vec.val;
        end
        if (isempty(u))
            sigmaPts = func(sigmaPts);
        else
            sigmaPts = func(sigmaPts,u);
        end
        vout.grad = sum(sigmaPts.grad .* Wm,2);
        vout.val = sum(sigmaPts.val .* Wm,2);
        res1.grad = sigmaPts.grad - vout.grad;
        res1.val = sigmaPts.val - vout.val;
    else
        vout = 0; res1 = 0; res2 = 0;
    end
end