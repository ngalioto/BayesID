function vout = propV(m,n,p,vec,mat,B,u)
    vout.grad = gradV(m,n,p,vec,mat);
    vout.val = mat.val*vec.val;
    if (nargin == 7)
        vout.val = vout.val + B.val*u;
        vout.grad = vout.grad + B.grad*u;
    end
end