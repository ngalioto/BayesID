% constructs the U matrix used for LS
function U = formU(u,N,nbar)
    [p,~] = size(u);
    idx2 = nbar:-1:1;
    U = zeros(nbar*p,N);
    for j = 1:N
        U(:,j) = reshape(u(:,idx2),[p*nbar,1]);
        idx2 = idx2 + 1;
    end
end