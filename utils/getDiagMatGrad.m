function grad = getDiagMatGrad(n,ind,mat,grad) %unconstrained variance
    for i = 1:n
        partInd = n*(ind(i)-1);
        grad(partInd + i,i) = mat(i,i);
    end
    grad = sparse(grad);
end