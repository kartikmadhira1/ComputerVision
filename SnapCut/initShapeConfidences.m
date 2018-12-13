
function ShapeConfidences = initShapeConfidences(LocalWindows, mask,ColorModels, WindowWidth, SigmaMin, A, fcutoff, R,colorConstant,param,nGauss)
% INITSHAPECONFIDENCES Initialize shape confidences.  ShapeConfidences is a struct you should define yourself.

ShapeConfidences=cell(length(LocalWindows));
for i=1:length(LocalWindows)
    %length(LocalWindows)
    x = LocalWindows(i,1);
    y = LocalWindows(i,2);
    patchBW=imcrop(mask,[x-(WindowWidth/2),y-(WindowWidth/2),WindowWidth,WindowWidth]);
    sigmaUpdate=SigmaMin+A*((ColorModels{i}-fcutoff).^R);
    
    if(ColorModels{i}<colorConstant)
        ShapeConfidences{i}=1-(exp((-bwdist(patchBW).^2)./(param)^2));
    else
        ShapeConfidences{i}=1-(exp((-1*bwdist(patchBW).^2)./(sigmaUpdate)^2));
    end

end
   
 
    
