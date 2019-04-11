
%% References:

% [1] https://cmsc426.github.io/2018/proj/p1/
% [2] https://in.mathworks.com/matlabcentral/fileexchange/4705-error_ellipse


%% Clean Slate

%% Clean Slate

close all; warning off;
clear all;
clc;

%% Switch to the current directory of mfile.

if(~isdeployed)
  cd(fileparts(which(mfilename)));
end

%% Load the Trained Data Values
load('trained_GMM_Data.mat');
pause(2)


%%  Move to the Base Samples Directory 
cd test_images/    % Switch to dir of the Data

%% Define a directory variable that has the path of the folder

directory   = '../test_images/';
sourceFiles = dir(fullfile(directory,'*.jpg'));

% Tried sorting the file names
% with Natsort addon which sorts the files names in proper order 
% as the input
sourceFiles = natsortfiles({sourceFiles.name});
fileCount   = numel(sourceFiles);   % To calculate the total number of files in the directory.

for inputFile=1:fileCount
    
    % Read the image
    currentImage  = imread(sourceFiles{inputFile});
    originalImage = imread(sourceFiles{inputFile});
    %reshape into the format n*n x 1
    currentImage = im2double(currentImage);
    currentImage = reshape(currentImage,640*480,3);
    currentImage = kGMM(scalingFactor,meanVec3d,covVec3d,K,currentImage);
    currentImage = reshape(currentImage,640,480);
    finalImage = currentImage > 17;
    originalImage(repmat(~finalImage,[1 1 3])) = 0;
    imshow(originalImage);
    pause(2)
    close
end

%% Change dir to original directory

cd ..


function post=kGMM(scalingFactor,meanVec3d,covVec3d,K,data)
    %final posterior is
    post=0;
    for i=1:K
        %the posteriors are in the order of e-100 and hence multiplying by
        %e+103
        post=post+(scalingFactor(i,1)*mvnpdf([data(:,1) data(:,2) data(:,3)],meanVec3d(:,:,i),covVec3d(:,:,i))*0.5*1e+143);
    end
    
end
