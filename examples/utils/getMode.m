function mode = getMode(samples)
    if (size(samples,2) > 1)
        N = size(samples,1);
        mode = zeros(N,1);
        for i = 1:N
            [f,xi] = ksdensity(samples(i,:));
            [~,ind] = max(f);
            mode(i) = xi(ind);
        end
    else
        mode = samples;
    end
end
