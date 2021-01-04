% Right half-normal probability distribution function
function p = rhnpdf(xin, mu, sigma)
    if (xin >= mu)
        p = 2*mvnpdf(xin,mu,sigma);
    else
        p = 0;
    end
end