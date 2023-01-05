% constructs the Y matrix
function Y = formY(y,N,nbar)
    [m,~] = size(y);
    Y = zeros(m,N);
    for j = 1:N
        Y(:,j) = y(:,j+nbar-1);
    end
end