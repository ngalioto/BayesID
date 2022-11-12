%https://homepages.inf.ed.ac.uk/imurray2/pub/16choldiff/choldiff.pdf
function [xout,err] = formSigmaPoints_grad(xin, Pin, n, ptot, lambda,indbwd,indfwd)
	[L,err] = chol(Pin.val,'lower');
    if (err == 0)
        In = eye(n);
        invL = In / L;
        gradReshape = reshape(Pin.grad*invL',[n,n*ptot]);
        gradReshape = invL*gradReshape(:,indfwd);
        invLdL = Phi(reshape(gradReshape(:,indbwd),[n*ptot,n]),n,ptot);
        gradReshape = reshape(invLdL,[n,n*ptot]);
        gradReshape = L*gradReshape(:,indfwd);
        dL = reshape(gradReshape(:,indbwd),[n*ptot,n]);
        scale = sqrt(n+lambda);
        Lscaled = scale*L;
        dLscaled = scale*dL;
        xout.val = zeros(n, 2*n+1);
        xout.grad = zeros(n*ptot, 2*n+1);
        xout.val(:, 1) = xin.val;
        xout.grad(:,1) = xin.grad;
        
        xout.val(:,2:n+1) = xin.val + Lscaled;
        xout.val(:,n+2:end) = xin.val - Lscaled;
        xout.grad(:,2:n+1) = xin.grad + dLscaled;
        xout.grad(:,n+2:end) = xin.grad - dLscaled;
    else
        xout = xin;
    end
end

function PhiA = Phi(A,n,ptot)
    nmat = ones(n);
    ii_phi = tril(nmat) - diag(0.5*diag(nmat));
    ii_phi = kron(ones(ptot,1),ii_phi);
    PhiA = A.*ii_phi;
end