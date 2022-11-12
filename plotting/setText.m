function setText(size,style)
if (nargin==1)
    style = 'latex';
end
    set(0,'defaultAxesFontSize',size)
    set(0, 'defaultLegendFontSize', size)
    set(0, 'defaultColorbarFontSize', size);

    set(0,'defaulttextinterpreter',style)
    set(0, 'defaultAxesTickLabelInterpreter',style)
    set(0, 'defaultLegendInterpreter', style)
    set(0, 'defaultColorbarTickLabelInterpreter',style)
end