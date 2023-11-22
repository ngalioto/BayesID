function y  = generateData(f, x0, t, H, sigmaR)
    [~, x] = ode45(@(t,x)f(x), t, x0);
    y = H*x' + sigmaR*randn(size(H*x'));
end