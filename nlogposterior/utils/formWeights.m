function [Wm, Wc] = formWeights(n, lambda, alpha, beta)
    Wm = zeros(1, 2*n+1);
    Wc = zeros(1, 2*n+1);

    Wm(1) = lambda / (n+lambda);
    Wm(2:end) = 1 / (2*(n + lambda));
    Wc(1) = lambda / (n+lambda) + 1 - alpha^2 + beta;
    Wc(2:end) = 1 / (2*(n+lambda));
end