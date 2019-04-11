%initial explore data
I=imread('68.jpg');
R=reshape(I,640*480,3);
R=cast(R,'single');
R=R/255.0;
size(R)

%read all the files in the directory
path=dir('*.jpg');
nFiles=length(path);
%vert stack to store R,G,B channels of entire dataset
imageStack=zeros(1,3)
for i=1:nFiles-5
    currImagePath=path(i).name;
    %read the image
    currImage=imread(currImagePath);
    %reshape into the format n*n x 1
    
    %convert data to a single precision value
    currImage= rgb2hsv(currImage);
    % Define thresholds for channel 1 based on histogram settings
    channel1Min = 0.010;
    channel1Max = 0.070;

    % Define thresholds for channel 2 based on histogram settings
    channel2Min = 0.563;
    channel2Max = 1.000;

    % Define thresholds for channel 3 based on histogram settings
    channel3Min = 0.000;
    channel3Max = 1.000;

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
end    

%remove the first row
imageStack(1,:)=[];
size(imageStack)

%get the cov and mean matrix of the whole dataset
covHSV=cov(imageStack)*10000
meanHSV=mean(imageStack)*10000
error_ellipse(covHSV);

%read the test images the other files
for i=1:nFiles
    currImagePath=path(i).name;
    %read the image
    currImage=imread(currImagePath);
    oriImage=imread(currImagePath);
    %reshape into the format n*n x 1
    currImage= rgb2hsv(currImage);
    currImage=im2double(currImage);
    currImage=reshape(currImage,640*480,3);
    currImage=mvnpdf([currImage(:,1) currImage(:,2) currImage(:,3)],meanHSV,covHSV)*0.5;
    currImage=reshape(currImage,640,480);
    finalImage=currImage>0.0000250;
    oriImage(repmat(~finalImage,[1 1 3])) = 0;
    imshow(oriImage);
    pause(1)
    close
end







