function [x_out] = simple_rnn_grad(ii_grad,x_in,ptot,param,ind1_param,n_x,n_in,n_out,n_nodes_per_layer)
    n_sigma = size(x_in.val,2);
    indA1 = (1:n_out*n_nodes_per_layer) + (ind1_param -1);
    A1 = reshape(param(indA1),[n_out,n_nodes_per_layer]);
    indA2 = indA1(end)+1:indA1(end) + n_nodes_per_layer*n_in;
    A2 = reshape(param(indA2),[n_nodes_per_layer,n_in]);
    indb2 = indA2(end)+1 : indA2(end) + n_nodes_per_layer;
    b2 = param(indb2);
    indA3 = indb2(end)+1 : indb2(end)+n_out*n_in;
    A3 = reshape(param(indA3),[n_out,n_in]);
    indb3 = indA3(end)+1 : indA3(end)+n_out;
    b3 = param(indb3);
    
    layer1_noact = A2*x_in.val + b2;
    layer1_withact = tanh(layer1_noact);
    x_out.val = A1*layer1_withact + A3*x_in.val + b3;
    
    dlayer1 = sech(layer1_noact).^2;
    A2_diag = zeros(n_nodes_per_layer*n_x,n_nodes_per_layer);
    idx = 1:n_nodes_per_layer;
    for i = 1:n_x
        A2_diag(idx,:) = diag(A2(:,i));
        idx = idx + n_nodes_per_layer;
    end
    gradx = reshape(A2_diag*dlayer1,[n_nodes_per_layer,n_x*n_sigma]);
    gradx = reshape(A1*gradx + kron(ones(1,n_sigma),A3(:,1:n_x)),[n_out*n_x,n_sigma]); 
%     gradx = A2'*A1dlayer1 + A3';
%     gradx = blk_kron(A2,A1dlayer1) + kron(ones(1,n_sigma),A3);
%     gradx = A1dlayer1*A2 + A3;
    
    gradp = zeros(n_out*ptot,n_sigma);
    In_out = eye(n_out);
    gradp((ind1_param-1)*n_out+1:indA1(end)*n_out,:) = reshape(kron(layer1_withact(:)',In_out),[n_out*numel(A1),n_sigma]);

    dlayer1_diag = zeros(n_nodes_per_layer,n_nodes_per_layer*n_sigma);
    xin_diag = zeros(n_sigma,n_sigma*n_in);
    idx_nodes = 1:n_nodes_per_layer;
    idx_in = 1:n_in;
    for i = 1:n_sigma
        dlayer1_diag(:,idx_nodes) = diag(dlayer1(:,i));
        xin_diag(i,idx_in) = x_in.val(:,i)';
        idx_nodes = idx_nodes + n_nodes_per_layer;
        idx_in = idx_in + n_in;
    end
    A1dlayer1_diag = reshape(A1*dlayer1_diag,[n_out*n_nodes_per_layer,n_sigma]);
    gradp(1+indA1(end)*n_out:indA2(end)*n_out,:) = reshape(A1dlayer1_diag*xin_diag,[n_out*numel(A2),n_sigma]);
    gradp(1+indA2(end)*n_out:indb2(end)*n_out,:) = A1dlayer1_diag;
    gradp(1+indb2(end)*n_out:indA3(end)*n_out,:) = reshape(kron(x_in.val(:)',In_out),[n_out*numel(A3),n_sigma]);
    gradp(1+indA3(end)*n_out:indb3(end)*n_out,:) = In_out(:).*ones(1,n_sigma);
    
    x_out.grad = gradp;
    for i = 1:n_sigma
        x_out.grad(:,i) = x_out.grad(:,i) + reshape(reshape(gradx(:,i),[n_out,n_x])*reshape(x_in.grad(:,i),[n_x,ptot]),[n_out*ptot,1]);
    end
    idx_grad = reshape(1:ptot*n_out', [n_out, ptot]);
    idx_grad = idx_grad(:,ii_grad);
    x_out.grad = x_out.grad(idx_grad(:),:);
end