function [ColorModels,foreGMM,backGMM] = initializeColorModels(IMG, Mask, MaskOutline, LocalWindows, BoundaryWidth, WindowWidth,colorConstant,param,nGauss)
% INITIALIZAECOLORMODELS Initialize color models.  ColorModels is a struct you should define yourself.
%
% Must define a field ColorModels.Confidences: a cell array of the color confidence map for each local window.

%{
Algorithm:
1. Take every window center, take the window size and get the binary patch.
2. For every window, create a background as well as foreground color gmm distribution.
3. Store
%}

ColorModels=cell(WindowWidth);
foreGMM=cell(WindowWidth);
backGMM=cell(WindowWidth);


for i = 1:length(LocalWindows);
    %length(LocalWindows)
    x = LocalWindows(i,1);
    y = LocalWindows(i,2);
    %% get patch data and independant RGB data
    
    patchImage=imcrop(IMG,[x-(WindowWidth/2),y-(WindowWidth/2),WindowWidth,WindowWidth]);
    patchImage=rgb2lab(patchImage);
    patchL=reshape(patchImage(:,:,1),(WindowWidth+1)*(WindowWidth+1),1);
    patchA=reshape(patchImage(:,:,2),(WindowWidth+1)*(WindowWidth+1),1);
    patchB=reshape(patchImage(:,:,3),(WindowWidth+1)*(WindowWidth+1),1);
    patchData=[patchL,patchA,patchB];
    
    %% create background and foreground data
    %create copies of this patch
    backPatch=patchImage;
    forePatch=patchImage;
    patchBW=imcrop(Mask,[x-(WindowWidth/2),y-(WindowWidth/2),WindowWidth,WindowWidth]);
    %backGround=patchImage.*repmat(patchBW,[1,1,3]);
%     subplot(1,2,1)
%     imshow(forePatch);hold on;
%     subplot(1,2,2)
%     imshow(backPatch);
%     hold off;
%     
    weird=patchBW;
    notWeird=~patchBW;
    patchLinearMaskFore=reshape(weird,(WindowWidth+1)*(WindowWidth+1),1);
    patchLinearMaskBack=reshape(notWeird,(WindowWidth+1)*(WindowWidth+1),1);

    Rfore=reshape(forePatch(:,:,1),(WindowWidth+1)*(WindowWidth+1),1);
    Gfore=reshape(forePatch(:,:,2),(WindowWidth+1)*(WindowWidth+1),1);
    Bfore=reshape(forePatch(:,:,3),(WindowWidth+1)*(WindowWidth+1),1);
     
    Rback=reshape(backPatch(:,:,1),(WindowWidth+1)*(WindowWidth+1),1);
    Gback=reshape(backPatch(:,:,2),(WindowWidth+1)*(WindowWidth+1),1); 
    Bback=reshape(backPatch(:,:,3),(WindowWidth+1)*(WindowWidth+1),1);
%     
    Rfore=Rfore(patchLinearMaskFore);
    Gfore=Gfore(patchLinearMaskFore);
    Bfore=Bfore(patchLinearMaskFore);
    
    Rback=Rback(patchLinearMaskBack);
    Gback=Gback(patchLinearMaskBack);
    Bback=Bback(patchLinearMaskBack);
    
   
%     
%     logicalFore=((Rfore~=0) & (Gfore~=0) & (Bfore~=0));
%     logicalBack=((Rback~=0) & (Gback~=0) & (Bback~=0));
     
    foreData=[Rfore Gfore Bfore];
    backData=[Rback Gback Bback];
    
  
    options = statset('MaxIter',200);
    gmmFore=gmdistribution.fit(foreData,nGauss,'RegularizationValue',0.1, 'Options', options);
    gmmBack=gmdistribution.fit(backData,nGauss,'RegularizationValue',0.1, 'Options', options);

    foreGMM{i}=gmmFore;
    backGMM{i}=gmmBack;
    %fit the patch data to the two gmm distributions
    Pfore=pdf(gmmFore,patchData);
    Pback=pdf(gmmBack,patchData);
    
    Pfore=reshape(Pfore,[(WindowWidth+1),(WindowWidth+1)]);
    Pback=reshape(Pback,[(WindowWidth+1),(WindowWidth+1)]);
    
    %the color confidence is pc(x)=p(x|F)/p(x|F)+p(x|B)
    Px=Pfore./(Pback+Pfore);
%     subplot(1,2,1);
%     imshow(Px);hold on
%     subplot(1,2,2);
%     imshow(patchImage);
%     hold off;

    
    weighFunc=exp((-1*bwdist(patchBW)^2)./((WindowWidth+1)/2)^2);
    weighSum=sum(weighFunc(:));
    patchSum=abs(patchBW-Px).*weighFunc;
    ColorModels{i}=1-(sum(sum(patchSum(:)))/sum(weighSum));
end
end