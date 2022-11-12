function [xout,err] = formGHSigmaPoints(xin, P, xi)
	[L,err] = chol(P,'lower');
    if (err == 0)
        xout = xin + L*xi;
    else
        xout = xin;
    end
end