% suppose we have matrices composed of p blocks stacked vertically
% bwd indices yield block transpose with reshape command
% fwd indices will transpose it back to vertical
function [xbwd,xfwd,ybwd,yfwd] = getIndices(p,dx,dy)
    ybwd0 = 0:dy:dy*(p-1);
    yfwd0 = 0:p:p*(dy-1);
    xbwd0 = 0:dx:dx*(p-1);
    xfwd0 = 0:p:p*(dx-1);
    ybwd = zeros(1,dy*p);
    yfwd = zeros(1,dy*p);
    xbwd = zeros(1,dx*p);
    xfwd = zeros(1,dx*p);
    for i = 1:dy
        ybwd(p*(i-1)+1:p*i) = ybwd0 + i;
    end
    for i = 1:dx
        xbwd(p*(i-1)+1:p*i) = xbwd0 + i;
    end
    for i = 1:p
        yfwd(dy*(i-1)+1:dy*i) = yfwd0 + i;
        xfwd(dx*(i-1)+1:dx*i) = xfwd0 + i;
    end
end