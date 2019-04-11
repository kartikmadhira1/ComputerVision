%% References:

% [1] https://cmsc426.github.io/2018/proj/p1/
% [2] https://in.mathworks.com/matlabcentral/fileexchange/4705-error_ellipse

%% Clean Slate

close all; warning off;
clear all;
clc;

%% Switch to the current directory of mfile.

if(~isdeployed)
  cd(fileparts(which(mfilename)));
end

%%  Move to the Base Samples Directory 
cd train_images/    % Switch to dir of the Data

%% Define a directory variable that has the path of the folder

directory   = '../train_images/';
sourceFiles = dir(fullfile(directory,'*.jpg'));

% Tried sorting the file names
% with Natsort addon which sorts the files names in proper order 
% as the input
sourceFiles = natsortfiles({sourceFiles.name});
fileCount   = numel(sourceFiles);   % To calculate the total number of files in the directory.

disp('Reading the Image files from Training DataSet ..');

imageStack = zeros(1,3);

for inputFile=1:fileCount
    
    fprintf('Input_image - %s \n',int2str(inputFile));
    filename = char(sourceFiles(inputFile));
    % Read the image
    currImage = imread(filename);
 
    %convert input image to HSV Color space
%     currImage= rgb2hsv(currImage);
    
    % Define thresholds for channel 1 based on histogram settings
    channel1Min = 156.000;
    channel1Max = 255.000;

    % Define thresholds for channel 2 based on histogram settings
    channel2Min = 0.000;
    channel2Max = 129.000;

    % Define thresholds for channel 3 based on histogram settings
    channel3Min = 39.000;
    channel3Max = 65.000;
    
    % Create mask based on chosen histogram thresholds
    sliderBW = (currImage(:,:,1) >= channel1Min ) & (currImage(:,:,1) <= channel1Max) & ...
        (currImage(:,:,2) >= channel2Min ) & (currImage(:,:,2) <= channel2Max) & ...
        (currImage(:,:,3) >= channel3Min ) & (currImage(:,:,3) <= channel3Max);

    % Initialize output masked image based on input image.
    maskedRGBImage = currImage;
    % Set background pixels where BW is false to zero.
    maskedRGBImage(repmat(~sliderBW,[1 1 3])) = 0;
    maskedRGBImage=reshape(maskedRGBImage,640*480,3);
    imageStack=vertcat(imageStack,maskedRGBImage);
    maskedImage = find((imageStack(:,1))>0 & (imageStack(:,2))>0 & (imageStack(:,3))>0);
    imageStack = imageStack(maskedImage,:);
end    

imageStack=cast(imageStack,'double');

%% Initialize the Number of Gaussians to mix 

K = 7;

%% Generate Random Scaling Factor(scalingFactor), Mean(meanVec3d) and Covariance(covVec3d)

scalingFactor = abs(randn(K,1));

% Generate the mean for the K gaussians initiated
% Preallocate memeory to store 3*3 K arrays for covariance and 1*3 K arrays
% for mean

covVec3d=zeros(3,3);
for i=1:K
    covVec3d(:,:,i) = cov(imageStack);
end

meanVec3d=zeros(1,3);
for i=1:K
    meanVec3d(:,:,i) = abs(rand(1,3))+10*rand(1,3);
end

%% Plot Error Ellipses
for i = 1:K
    error_ellipse(covVec3d(:,:,i),meanVec3d(:,:,i));
    hold on
end
hold off
pause(5)
close

%%

latVarlen = size(imageStack);
prevMean  = zeros(1,3);
max_iters = 50;
% threshold = 0.5;
for j = 1: max_iters  
    
    %---------------------E-step-----------------------------------

    alpha_numerator      = [];
    for i = 1:K
        dummy_numerator = scalingFactor(i,1) * mvnpdf ( imageStack , meanVec3d(:,:,i) , covVec3d(:,:,i));
        alpha_numerator  = horzcat( alpha_numerator , dummy_numerator );
    end
    
    alpha_denominator = sum(alpha_numerator,2);
    alpha             = alpha_numerator ./ alpha_denominator;
    alphasum          = sum(alpha,1);
    prevMean          = meanVec3d;

    %---------------------M -Step-----------------------------------
    dummyVec3d=zeros(3,3);
    
    for i=1:K
        meanVec3d(:,:,i)=(alpha(:,i)'*(imageStack))./(sum(alpha(:,i)));
        mystery = imageStack-meanVec3d(:,:,i); 
        dummyVec3d(:,:,i)=zeros(3,3);
        for iter = 1:numel(alpha(:,1))
            dummyVec3d(:,:,i)= dummyVec3d(:,:,i) + (alpha(iter,i)* (mystery(iter,:)'*mystery(iter,:)));
        end  
        
        covVec3d(:,:,i) = dummyVec3d(:,:,i) / alphasum(1,i);
    end
    scalingFactor = sum(alpha,1)/length(alpha);
    scalingFactor = scalingFactor';  
end

datafileToSave = '../trained_GMM_Data.mat';
save(datafileToSave);

function M = randCov(N)
d = 10*rand(N,1); % The diagonal values
t = triu(bsxfun(@min,d,d.').*rand(N),1); % The upper trianglar random values
M = diag(d)+t+t.'; % Put them together in a symmetric matrix
end


function ind=matInd(i,K)
if mod(i,K)==0
    ind=3;
else
    ind=mod(i,K);
end

end
