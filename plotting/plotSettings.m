function plotSettings(fontSize,lineWidth,style)
    if (nargin==2)
        style = 'latex';
    end
    set(0,'DefaultLineLineWidth',lineWidth);
    
    set(0,'defaultAxesFontSize',fontSize)
    set(0, 'defaultLegendFontSize', fontSize)
    set(0, 'defaultColorbarFontSize', fontSize);

    set(0,'defaulttextinterpreter',style)
    set(0, 'defaultAxesTickLabelInterpreter',style)
    set(0, 'defaultLegendInterpreter', style)
    set(0, 'defaultColorbarTickLabelInterpreter',style)
end