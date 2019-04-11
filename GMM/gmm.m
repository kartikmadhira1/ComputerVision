% %{
% 1.Initialize the number of the clusters to create, in this case the number
% of gaussian models to replicate the color model of the thresholded values.
% 2.Intitialize mean and covariance randomly for these K gaussian models. 3.
% E step - Calculate the probability of each point in the dataset with each
% of the initiated gaussian models and check for the probabilities. The one
% with the higher probability gets higher weight. 4. M step - update the
% values of the mean and covariance of each of these models 5. Iteratively do
% steps 4 and 5 to converge to a value.
% %}

%read all the files in the directory
path=dir('*.jpg');
nFiles=length(path);
%vert stack to store R,G,B channels of entire dataset
imageStack=zeros(1,3);
for i=1:nFiles-5
    currImagePath=path(i).name
    %read the image
    currImage=imread(currImagePath);
    %reshape into the format n*n x 1
    %convert data to a single precision value
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
    ii1 = find((imageStack(:,1))>0 & (imageStack(:,2))>0 & (imageStack(:,3))>0);
    imageStack = imageStack(ii1,:);
end    

%remove the first row
imageStack=cast(imageStack,'double')
size(imageStack);

%get the cov and mean matrix of the whole dataset
covHSV=cov(imageStack)
%this is the mean near to which it should converge
meanHSV=mean(imageStack)

%initialize the number of gaussians to mix 
K=8;
%generate random ui, meani and covi
u=abs(rand(K,1));
%generate the means for the K gaussians initiated
%preallocate memeory to store 3*3 K arrays for covariance and 1*3 K arrays
%for mean
covVector=cell(K,1);
latMean=cell(K,1);
%%
covVec3d=zeros(3,3);
for i=1:K
    covVec3d(:,:,i)=cov(imageStack)
end

meanVec3d=zeros(1,3);
for i=1:K
    meanVec3d(:,:,i)=abs(rand(1,3))+10*rand(1,3);
end

latVarlen=size(imageStack);
latVarlen(1,1);
lastMean=zeros(1,3);
for j=1:50
    %initialize an empty array to store the probabilty values
    %---------------------E-step-----------------------------------
    latVar=zeros(latVarlen(1,1),1);
    latVar(:,1)=[];
    for i=1:K
        latVar=horzcat(latVar,u(i,1)*mvnpdf(imageStack,meanVec3d(:,:,i),covVec3d(:,:,i)));
    end
    psum=sum(latVar,2);
    %dividing each of the entries in the latent prob by the total sum
    alpha=latVar./psum;
    size(alpha);
    alphasum=sum(alpha,1);
    lastMean=meanVec3d;
    
    %---------------------M -Step-----------------------------------
    %%
    temp=[];
    dummyVec3d=zeros(3,3);
    ddd=zeros(3,3);
%     covVector=cell(K,1);
    for i=1:K
        meanVec3d(:,:,i)=(alpha(:,i)'*(imageStack))./(sum(alpha(:,i)));
        mystery = imageStack-meanVec3d(:,:,i); 
        dummyVec3d(:,:,i)=zeros(3,3);
        for iter = 1:numel(alpha(:,1))
            dummyVec3d(:,:,i)= dummyVec3d(:,:,i) + (alpha(iter,i)* (mystery(iter,:)'*mystery(iter,:)));
        end  
       
%         covVector{i}=zeros(3,3);
        covVec3d(:,:,i) = dummyVec3d(:,:,i) ./ alphasum(1,i);
        %covVector{i}=num2cell(bsxfun(@rdivide,cell2mat(dummy_sum(i)), sum(alpha(:,i))));
%         cell2mat(covVector{i})=cell2mat(dummy_sum{i})/sum(alpha(:,i));
        %covVector{i}=(transpose(alpha(:,i)).*transpose((imageStack)-latMean{i}))*((imageStack)-latMean{i})./sum(alpha(:,i))*100;
    end
  
    u=sum(alpha,1)/length(alpha);
    u=u';
    norm(meanVec3d(:,:,1)-lastMean(:,:,1))
end

function M=randCov(N)
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
