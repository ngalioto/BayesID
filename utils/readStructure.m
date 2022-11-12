% mat is the default array (matrix or vector)

% structure is an array of booleans of the same size as mat. If an element
% is set to true, then it is learnable. Otherwise it takes the value in mat

% ind is the indices of the parameters on which the array depends

% ptot is the number of total parameters. If empty, the gradient field is
% not created
function M = readStructure(mat,structure,ind,ptot)
    [m,n] = size(mat);
    if (nargin == 4)
        grad = getMatrixGrad(m,n,structure,ind,ptot);
        M = @(theta)struct('val',formMatrix(mat,structure,theta(ind)),'grad',grad);
    else
        M = @(theta)struct('val',formMatrix(mat,structure,theta(ind)));
    end
end

function mat = formMatrix(mat, structure, theta)
    mat(structure) = theta;
end

function grad = getMatrixGrad(m,n,structure,ind,ptot)
    idx = 1:m*n;
    idx = idx(structure(:));
    num = length(idx);
    indices = zeros(num,2);
    [row,col] = ind2sub([m,n],idx);
    indices(:,1) = m*(ind-1) + row;
    indices(:,2) = col;
    grad = sparse(indices(:,1), indices(:,2), ones(num,1), m*ptot, n);
end