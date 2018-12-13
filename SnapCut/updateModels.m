
function [mask, LocalWindows, ColorModels, ShapeConfidences] = ...
    updateModels(...
        NewLocalWindows, ...
        LocalWindows, ...
        CurrentFrame, ...
        foreGMM, ...
        backGMM, ...
        warpedMask, ...
        warpedMaskOutline, ...
        WindowWidth, ...
        ColorModels, ...
        ShapeConfidences, ...
        ProbMaskThreshold, ...
        fcutoff, ...
        SigmaMin, ...
        R, ...
        A, ...
        colorConstant, ...
        param, ...
        nGauss ...
    )

% UPDATEMODELS: update shape and color models, and apply the result to generate a new mask.
% Feel free to redefine this as several different functions if you prefer.

%{
Color updation steps:
1. Shape model Ms updated = Warped local window
2. New GMM of foreground model = sampling pixels in warped whose shape
model > 0.75.
3. New GMM of background model = sampling pixels in warped whose shape
model > 0.25
%}
    %get the shapeConfidences in this win  dow first
Mask=zeros(size(warpedMask));
integP=cell(length(LocalWindows));
newShapeConfidences = initShapeConfidences(NewLocalWindows, warpedMask,ColorModels, WindowWidth, SigmaMin, A, fcutoff, R,colorConstant,param,nGauss);
for i=1:length(LocalWindows)
    %go to each window, gather all the pixels that have a higher threshold
    %from the updated shapeconfidences.
    
    %%%DO NOT FORGET TO convert to lab space.
    x = round(NewLocalWindows(i,1));
    y = round(NewLocalWindows(i,2));
    
%     if(isnan(x)==1 | isnan(y)==1)
%         x=prev_x;
%         y=prev_y;
%     end
%     
    
    %get the binary mask of the window
    patchBW=imcrop(warpedMask,[x-(WindowWidth/2),y-(WindowWidth/2),WindowWidth,WindowWidth]);
   
    %multiply with mask to get the foreground and background data
    %seperately.
    newShapeFore=newShapeConfidences{i}.*patchBW;
    newShapeBack=newShapeConfidences{i}.*(~patchBW);
    
    %get the coors of the windows where its large.
    [rF,cF]=find(newShapeFore>0.75);
    
    [rB,cB]=find(newShapeBack>0.25);
    %now apply this to the converted image frame
    warpedWindow=imcrop(CurrentFrame,[x-(WindowWidth/2),y-(WindowWidth/2),WindowWidth,WindowWidth]);
    
    if(isnan(x)==1 | isnan(y)==1)
        patchBW=ones((WindowWidth+1),(WindowWidth+1));
        warpedWindow=ones((WindowWidth+1),(WindowWidth+1),3);
    end
    
    %convert to LAB colorspace and get the LAB color datas
    warpedWindow=rgb2lab(warpedWindow);    
    windowLABFore = impixel(warpedWindow,cF,rF);
    windowLABBack = impixel(warpedWindow,cB,rB);
    size1=size(windowLABFore);
    if(size1(1,1)<4)

        gmmFore=gmdistribution.fit(zeros(4,3),1,'RegularizationValue',0.1);
    else
        gmmFore=gmdistribution.fit([windowLABFore(:,1),windowLABFore(:,2),windowLABFore(:,3)],1,'RegularizationValue',0.1);
       
    end
    size2=size(windowLABBack);
    if(size2(1,1)<4)

        gmmBack=gmdistribution.fit(zeros(4,3),nGauss,'RegularizationValue',0.1);
    else
        gmmBack=gmdistribution.fit([windowLABBack(:,1),windowLABBack(:,2),windowLABBack(:,3)],nGauss,'RegularizationValue',0.1);
    end
    
    %compare the probs of updated and the history models of only the
    %background
    %updated and history models are the same.
    pForeUpdated=pdf(foreGMM{i},[reshape(warpedWindow(:,:,1),(WindowWidth+1)*(WindowWidth+1),1),reshape(warpedWindow(:,:,2),(WindowWidth+1)*(WindowWidth+1),1),reshape(warpedWindow(:,:,3),(WindowWidth+1)*(WindowWidth+1),1)]);
    %pForeHistory=pdf(foreGMM{i},[reshape(warpedWindow(:,:,1),(WindowWidth+1)*(WindowWidth+1),1),reshape(warpedWindow(:,:,2),(WindowWidth+1)*(WindowWidth+1),1),reshape(warpedWindow(:,:,3),(WindowWidth+1)*(WindowWidth+1),1)]);

    pBackHistory=pdf(backGMM{i},[reshape(warpedWindow(:,:,1),(WindowWidth+1)*(WindowWidth+1),1),reshape(warpedWindow(:,:,2),(WindowWidth+1)*(WindowWidth+1),1),reshape(warpedWindow(:,:,3),(WindowWidth+1)*(WindowWidth+1),1)]);
    pBackUpdated= pdf(gmmBack,[reshape(warpedWindow(:,:,1),(WindowWidth+1)*(WindowWidth+1),1),reshape(warpedWindow(:,:,2),(WindowWidth+1)*(WindowWidth+1),1),reshape(warpedWindow(:,:,3),(WindowWidth+1)*(WindowWidth+1),1)]);

    pForeUpdated=reshape(pForeUpdated,[(WindowWidth+1),(WindowWidth+1)]);
    pForeHistory=reshape(pForeUpdated,[(WindowWidth+1),(WindowWidth+1)]);
    pBackUpdated=reshape(pBackUpdated,[(WindowWidth+1),(WindowWidth+1)]);
    pBackHistory=reshape(pBackHistory,[(WindowWidth+1),(WindowWidth+1)]);

    pUpdated=pForeUpdated./(pBackUpdated+pForeUpdated);
    pHistory=pForeUpdated./(pBackHistory+pForeUpdated);
%     subplot(1,2,1)
%     imshow(pUpdated);hold on;
%     subplot(1,2,2);
%     imshow(pHistory);
%     hold off
    updatedCount=length(find(pUpdated>0.9));
    historyCount=length(find(pHistory>0.9));
    %take the one with the higher count 
    %%%CHANGE BACK TO P(X)=HISTORY MODEL
    
    
    Px=pHistory;
%     if(updatedCount>historyCount)
%         Px=pUpdated;
%     else
%         Px=pHistory;
%     end
    weighFunc=exp((-1*bwdist(patchBW)^2)/((WindowWidth+1)/2)^2);
    weighSum=sum(weighFunc(:));
    patchSum=abs(patchBW-Px).*weighFunc;
    ColorModels{i}=1-(sum(sum(patchSum(:)))/sum(weighSum));    
    
    sigmaUpdate=SigmaMin+A*((ColorModels{i}-fcutoff)^R);
    %find wherever the colorconfidences are less than a certain point
    %ShapeConfidences(find(ColorModels{i}<=0.5))=1-exp((-bwdist(patchBW)^2)/SigmaMin); 
    %ShapeConfidences(find(ColorModels{i}>=0.5))=sigmaUpdate;
    if(ColorModels{i}<colorConstant)
        ShapeConfidences{i}=1-(exp((-bwdist(patchBW)^2)./(SigmaMin)^2));
    else
        ShapeConfidences{i}=1-(exp((-bwdist(patchBW)^2)./(sigmaUpdate)^2));
    end
    
%      ShapeConfidences{i}=1-(exp((-bwdist(patchBW)^2)/(WindowWidth/4)^2));
 
    integP{i}=ShapeConfidences{i}.*patchBW+(1-ShapeConfidences{i}).*Px;
    subplot(1,4,1);
    imshow(integP{i}); hold on
    subplot(1,4,2);

    imshow(ShapeConfidences{i});
    subplot(1,4,3);
    imshow(Px);
    subplot(1,4,4);
    imshow(imcrop(CurrentFrame,[x-(WindowWidth/2),y-(WindowWidth/2),WindowWidth,WindowWidth]));
    hold off;
      prev_x=x;
      prev_y=y;
end

centerVals = zeros(WindowWidth+1,WindowWidth+1);
centerVals(WindowWidth/2 ,WindowWidth/2) = 1;
centerVals = bwdist(centerVals);
C= 0.1;
outerFore= zeros(size(rgb2gray( CurrentFrame)));
outerForeCell = {};

for i=1:length(NewLocalWindows)
    Fore = zeros(size(rgb2gray( CurrentFrame)));
    X = round(NewLocalWindows(i,1));
    Y = round(NewLocalWindows(i,2));
    if(isnan(X)==1 | isnan(Y)==1)
        X=prev_x;
        Y=prev_y;
    end
    Pc = integP{i};
    dist = (centerVals + C).^-1;
    Pk=(Pc.*dist)./dist;
    Fore(Y-WindowWidth/2:Y+WindowWidth/2,X-WindowWidth/2:X+WindowWidth/2) = Pk;
    outerForeCell{i} = Fore;
    prev_x=X;
    prev_y=Y;
end

for i = 1:numel(outerForeCell)
   outerFore = outerFore + outerForeCell{i}; 
end

grayFore = mat2gray(warpedMask);
t = WindowWidth/2;
for i=1:length(NewLocalWindows)
   X = round(NewLocalWindows(i,1));
   Y = round(NewLocalWindows(i,2));
    if(isnan(X)==1 | isnan(Y)==1)
        X=prev_x;
        Y=prev_y;
    end
   grayFore(Y-t:Y+t,X-t:X+t) = 0;      
    prev_x=X;
    prev_y=Y;
end

outerFore(isnan(outerFore)) = 0;

outerFore = outerFore + grayFore;
threshMask= outerFore > 0.9;
mask=threshMask;
LocalWindows=NewLocalWindows;
end
