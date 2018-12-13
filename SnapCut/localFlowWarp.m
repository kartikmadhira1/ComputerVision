

function [NewLocalWindows] = localFlowWarp(WarpedPrevFrame, CurrentFrame, LocalWindows, Mask, Width)
WindowWidth=Width;
%get the flow vectors and average them out
opticFlow = opticalFlowFarneback();
frameGray = rgb2gray(WarpedPrevFrame);
flow = estimateFlow(opticFlow,frameGray); 
flow = estimateFlow(opticFlow,rgb2gray(CurrentFrame));
imshow(WarpedPrevFrame) 
hold on
plot(flow,'DecimationFactor',[5 5],'ScaleFactor',3)
hold off 
% NewLocalWindows=ones(size(LocalWindows));
%get the values of flow vectors of each window and then add the averaged
%flow vector and add it to windowcentres

for i=1:length(LocalWindows)
    %mask out the local foreground
    x = LocalWindows(i,1);
    y = LocalWindows(i,2);
    fullVx=imcrop(flow.Vx,[x-(WindowWidth/2),y-(WindowWidth/2),WindowWidth,WindowWidth]);
    fullVy=imcrop(flow.Vy,[x-(WindowWidth/2),y-(WindowWidth/2),WindowWidth,WindowWidth]);
    patchBW=imcrop(Mask,[x-(WindowWidth/2),y-(WindowWidth/2),WindowWidth,WindowWidth]);
    imshow(patchBW)
    fullVx=fullVx.*patchBW;
    fullVy=fullVy.*patchBW;
    Vx=(mean(mean(nonzeros(fullVx),2,'omitnan'),1,'omitnan'));
    Vy=(mean(mean(nonzeros(fullVy),2,'omitnan'),1,'omitnan'));
    
    if(isnan(fullVx)==1)
        disp('NAN IS PRESENT');
    end
%     if(Vy>abs(0.2))
%           imshow(Mask(y+Vy-(WindowWidth/2):y+Vy+(WindowWidth/2),x+Vx-(WindowWidth/2):x+Vx+(WindowWidth/2)));
% 
%     end
    NewLocalWindows(i,1)=x+round(Vx);
    NewLocalWindows(i,2)=y+round(Vy);
    
    
end
end
