function [XI, W] = formGHWeights(n, p)
    c = 1:(p-1);
    J = diag(sqrt(c), 1);
    J = J + J';
    [w,xi] = eig(J);
    xi = diag(xi); w = w(1,:).^2;
    idx = cell(1,n);
    [idx{:}] = ndgrid(1:p);
    XI = zeros(n,p^n);
    W = ones(1,p^n);
    for i = 1:n
        XI(i,:) = xi(idx{i}(:));
        W = W .* w(idx{i}(:));
    end
end