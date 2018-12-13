% MyRotobrush.m  - UMD CMSC426, Fall 2018
% This is the main script of your rotobrush project.
% We've included an outline of what you should be doing, and some helful visualizations.
% However many of the most important functions are left for you to implement.
% Feel free to modify this code as you see fit.

% Some parameters you need to tune:
WindowWidth = 30;  
ProbMaskThreshold = 1; 
NumWindows=40; 
BoundaryWidth = 1;

colorConstant=0.5;
param=WindowWidth/2;
nGauss=2;


% Load images:
fpath = '../input/frames/Frames1/';
files = dir(fullfile(fpath, '*.jpg'));
imageNames = zeros(length(files),1);
images = cell(length(files),1);

for i=1:length(files)
    imageNames(i) = str2double(strtok(files(i).name,'.jpg'));
end

imageNames = sort(imageNames);
imageNames = num2str(imageNames);
imageNames = strcat(imageNames, '.jpg');

for i=1:length(files)
    images{i} = im2double(imread(fullfile(fpath, strip(imageNames(i,:)))));
end

% NOTE: to save time during development, you should save/load your mask rather than use ROIPoly every time.
mask=roipoly(images{1});

% Sample local windows and initialize shape+color models:
[mask_outline, LocalWindows] = initLocalWindows(images{1},mask,NumWindows,WindowWidth,false);


[ColorModels,foreGMM,backGMM] = ...
    initColorModels(images{1},mask,mask_outline,LocalWindows,BoundaryWidth,WindowWidth,colorConstant,param,nGauss);

% You should set these parameters yourself:
fcutoff = 0.85;
SigmaMin = 2;
SigmaMax = WindowWidth;
R = 2;
A = (SigmaMax-SigmaMin)/((1-fcutoff)^R);
ShapeConfidences = ...
    initShapeConfidences(LocalWindows,mask,ColorModels,...
    WindowWidth, SigmaMin, A, fcutoff, R,colorConstant,param,nGauss);



%%% MAIN LOOP %%%
% Process each frame in the video.
for prev=1:length(LocalWindows)
    curr = prev+1;
    fprintf('Current frame: %i\n', curr)
    
    [warpedFrame, warpedMask, warpedMaskOutline, warpedLocalWindows] = calculateGlobalAffine(images{prev}, images{curr}, mask, LocalWindows);

    [NewLocalWindows] = ...
        localFlowWarp(warpedFrame, images{curr}, warpedLocalWindows,warpedMask,WindowWidth);
    
    
    [ ...
        mask, ...
        LocalWindows, ...
        ColorModels, ...
        ShapeConfidences, ...
    ] = ...
    updateModels(...
        NewLocalWindows, ...
        LocalWindows, ...
        images{curr}, ...
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
    );
    
    mask_outline = bwperim(mask,4);


    filename = strcat(num2str(prev),'.jpg');
    filename = strcat('../new/',filename);
    imwrite(imoverlay(images{curr}, boundarymask(mask_outline,4), 'red'),filename);
    

%}
end