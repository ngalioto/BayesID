function x_out = simple_res_net(x_in,param,n_in,n_out,num_nodes)
    indA1 = 1:n_out*num_nodes;
    A1 = reshape(param(indA1),[n_out,num_nodes]);
    indA2 = indA1(end)+1:indA1(end) + num_nodes*n_in;
    A2 = reshape(param(indA2),[num_nodes,n_in]);
    indb2 = indA2(end)+1 : indA2(end) + num_nodes;
    b2 = param(indb2);
    indA3 = indb2(end)+1 : indb2(end)+n_out*n_in;
    A3 = reshape(param(indA3),[n_out,n_in]);
    indb3 = indA3(end)+1 : indA3(end)+n_out;
    b3 = param(indb3);
    
    layer1_noact = A2*x_in + b2;
    layer1_withact = tanh(layer1_noact);
    x_out = A1*layer1_withact + A3*x_in + b3;
end