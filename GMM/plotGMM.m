load('trained_GMM_Data.mat');
%% Plot Error Ellipses
for i = 1:K
    error_ellipse(covVec3d(:,:,i),meanVec3d(:,:,i));
    hold on
end
hold off
pause(50)
close