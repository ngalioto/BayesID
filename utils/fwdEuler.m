% Time propagator using the forward Euler method
function xout = fwdEuler(xin, f, N, DT)
    xout = xin;
    for i = 1:N
        xout = xout + f(xout)*DT;
    end
end