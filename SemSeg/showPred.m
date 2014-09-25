function showPred( cls, vals, params, trImg, trLabels, teImg )
%SHOWPRED Summary of this function goes here
%   Detailed explanation goes here

%% init
nCls = size(vals, 2) - 1;

sampleMask = params.feat.sampleMask;
[rows, cols] = find(sampleMask);
lineCols = colormap('lines');
cls = reshape(cls, [numel(unique(rows)) numel(unique(cols))]);

%% show
figure(3000); clf;
subplot(1, 2, 1);
imshow(trImg);
hold on;
layerImg = zeros(size(trImg));
for cInd=1:nCls
    layerImg_c = zeros(size(trImg));
    layerImg_c(repmat(trLabels.cls == cInd, [1 1 3])) = 1;
    layerImg = layerImg + bsxfun(@times, layerImg_c, reshape(lineCols(cInd, :), [1 1 3]));
end
h = imagesc(layerImg);
set(h, 'AlphaData', 0.7);
hold off;
title('training regions');

subplot(1, 2, 2);
imshow(teImg);
hold on;
layerImg = zeros(size(cls, 1), size(cls, 2), 3);
for cInd=1:nCls
    layerImg_c = zeros(size(cls, 1), size(cls, 2), 3);
    layerImg_c(repmat(cls == cInd, [1 1 3])) = 1;
    layerImg = layerImg + bsxfun(@times, layerImg_c, reshape(lineCols(cInd, :), [1 1 3]));
end
h = imagesc(layerImg);
set(h, 'AlphaData', 0.7);
hold off;
title('Predicted regions');


% figure(3000); clf;
% for cInd=1:nCls
%     figure(3000); 
%     subplot(nCls, 2, (cInd-1)*2 + 1);
%     imshow(trImg);
%     % rectangle('Position', rect, 'EdgeColor', 'r');
%     % title('training regions represented by the bounding box');
%     hold on;
%     heatMap = trLabels.cls;
%     heatMap(heatMap ~= cInd) = 0;
%     heatMap(heatMap == cInd) = 255;
%     h = imagesc(heatMap);
%     set(h, 'AlphaData', 0.7);
%     hold off;
%     title('training regions');
% 
%     subplot(nCls, 2, (cInd-1)*2 + 2);
%     imshow(teImg);
%     hold on;
%     heatMap = zeros(size(sampleMask));
%     heatMap(sampleMask) = reshape(cls, [numel(unique(rows)) numel(unique(cols))]);
%     heatMap(heatMap ~= cInd) = 0;
%     heatMap(heatMap == cInd) = 255;
%     h = imagesc(heatMap);
%     set(h, 'AlphaData', 0.7);
%     hold off;
%     title('Predicted regions');
% end


end

