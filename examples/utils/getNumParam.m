function numParam = getNumParam(nVars, polyorder, usesine)
    numParam = factorial(polyorder+nVars) / ...
        (factorial(polyorder)*factorial(nVars-1)) + nVars*20*usesine;
end