
function [WarpedFrame, WarpedMask, WarpedMaskOutline, WarpedLocalWindows] = calculateGlobalAffine(IMG1,IMG2,Mask,Windows)
% CALCULATEGLOBALAFFINE: finds affine transform between two frames, and applies it to frame1, the mask, and local windows.
%get the first image, mask it and get the feature points.
img=IMG1;
mask=Mask;
%Inew = img.*repmat(mask,[1,1,3]);
Inewgray=rgb2gray(img);

Inewgray2=rgb2gray(IMG2);

pts1 = detectSURFFeatures(Inewgray,'MetricThreshold',200);
pts2 = detectSURFFeatures(Inewgray2,'MetricThreshold',200);

[ft1,vpoints1] = extractFeatures(rgb2gray(img), pts1);
[ft2,vpoints2] = extractFeatures(rgb2gray(IMG2),pts2);

%imshow(img1);
% hold on
% %plot(pts1.Location(:,1),pts1.Location(:,2),'.', 'Color', 'r');
% hold off
% 
 idxpair = matchFeatures(ft1,ft2);
 matchedPoints1 = vpoints1(idxpair(:, 1), :);
 matchedPoints2 = vpoints2(idxpair(:, 2), :);

%Estimage Geometric Transform
tform = estimateGeometricTransform(matchedPoints1.Location, ...
    matchedPoints2.Location, 'affine');
%indexPairs = matchFeatures(f1,f2);
%showMatchedFeatures(Inew,IMG2,vpts1(indexPairs(:, 1)),vpts2(indexPairs(:, 2)));

%tform = estimateGeometricTransform(vpts2(indexPairs(1:7, 2)),vpts1(indexPairs(1:7, 1)),'affine');

WarpedFrame=imwarp(img,tform,'OutputView', imref2d( size(img)));
WarpedMask=imwarp(mask,tform,'OutputView', imref2d( size(img)));
imshow(WarpedMask);
%updating the values for the all the windows
WarpedLocalWindows=round(transformPointsForward(tform,Windows));

WarpedMaskOutline=bwperim(WarpedMask,4);
end


