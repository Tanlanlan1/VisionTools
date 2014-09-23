function showPred( cls, vals, params, trImg, trLabels, teImg )
%SHOWPRED Summary of this function goes here
%   Detailed explanation goes here
sampleMask = params.feat.sampleMask;
[rows, cols] = find(sampleMask);

figure(1); clf;
subplot(1, 2, 1);
imshow(trImg);
% rectangle('Position', rect, 'EdgeColor', 'r');
% title('training regions represented by the bounding box');
hold on;
heatMap = trLabels.cls;
heatMap(heatMap ~= 1) = 0;
heatMap(heatMap == 1) = 255;
h = imagesc(heatMap);
set(h, 'AlphaData', 0.7);
hold off;
title('training regions');

subplot(1, 2, 2);
imshow(teImg);
hold on;
heatMap = zeros(size(sampleMask));
heatMap(sampleMask) = reshape(cls, [numel(unique(rows)) numel(unique(cols))]);
heatMap(heatMap ~= 1) = 0;
heatMap(heatMap == 1) = 255;
h = imagesc(heatMap);
set(h, 'AlphaData', 0.7);
hold off;
title('Predicted regions');



end

