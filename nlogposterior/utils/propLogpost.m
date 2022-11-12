function nlp = propLogpost(m,ptot,nlp,S,Sinv,v,indbwd,indfwd)
    vS = v.val'*Sinv.val;
    traceS = zeros(ptot,1);
    temp = reshape(S.grad, [m,m*ptot]);
    temp = Sinv.val*temp(:,indfwd);
    temp = reshape(temp(:,indbwd), [m*ptot,m]);
    for i = 1:ptot
        for j = 1:m
            traceS(i) = traceS(i) + temp(m*(i-1)+j,j);
        end
    end
    nlp.grad = nlp.grad + 0.5*traceS + ...
        (vS*reshape(v.grad,[m,ptot]))' + ...
        0.5*(v.val'*reshape(Sinv.grad*v.val,[m,ptot]))';
    nlp.val = nlp.val + 0.5*log(det(S.val)) + 0.5*(vS*v.val);
end