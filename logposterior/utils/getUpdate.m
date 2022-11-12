% function [m,P] = getUpdate(m,P,PH,S,v)
function [m,P] = getUpdate(m,P,PH,Sinv,Sinvv)
%     K = HP'/S;
%     m = m + K*v;
%     P = P - K*HP;
    m = m + PH*Sinvv;
    P = P - (PH*Sinv*PH');
    P = (P+P') / 2;
end