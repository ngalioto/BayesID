% Adds nuggets of increasing magnitude to a covariance matrix until the 
% matrix is positive semi-definite.
function [Pout,sqrtPout] = conditionCov(Pin,epsilon)
    n = size(Pin,1);
    if (nargin>1 && epsilon>0)
        i = log10(epsilon);
    else
        i = -9;
    end
    pd = -1;
    while (pd ~= 0)
        Pout = Pin + 10^i*eye(n);
        [sqrtPout, pd] = chol(Pout,'lower');
        i = i + 1;
    end
end