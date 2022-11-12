% Samples banana distribution
close all;
clear;
clc;

postC = [1 0.9; 0.9 1];
invpostC = inv(postC);
evalLP = @(x)evalLogPost(x,invpostC);
evalNLP = @(x)evalNLP_grad(x,invpostC);
options = optimoptions('fminunc','SpecifyObjectiveGradient',true);
[map, ~,~,~,~, hess] = fminunc(evalNLP, zeros(2,1), options);

N = 1e5;
propC = inv(hess);
[samples,acc,propC] = DRAM(map,propC,N,evalLP);


function p = evalLogPost(x,invC)
	delta = [x(1);x(2)+x(1)^2+1];
    p = -0.5*delta'*invC*delta;
end

function [nlp,g] = evalNLP_grad(x,invC)
    nlp = -evalLogPost(x,invC);
    g = [1 2*x(1); 0 1]*invC*[x(1);x(2)+x(1)^2+1];
end