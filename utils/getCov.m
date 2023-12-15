% Creates a diagonal, positive-definite matrix
% Parameters are unconstrained and passed through an exponential to enforce
% positivity

function Q = getCov(n,ind,ptot)
    Qmat = @(theta)diag(exp(theta(ind)));
    if (~isempty(ptot))
        grad = zeros(n*ptot,n);
        grad = @(theta)getDiagMatGrad(n,ind,Qmat(theta),grad);
        Q = @(theta)struct('val',Qmat(theta),'grad',grad(theta));
    else
        Q = @(theta)struct('val',Qmat(theta));
    end
end