function showPred( cls, vals, params, trImg, trLabels, teImg )
%SHOWPRED Summary of this function goes here
%   Detailed explanation goes here

%% init
nCls = size(vals, 2) - 1;

sampleMask = params.feat.sampleMask;
% [rows, cols] = find(sampleMask);
lineCols = colormap('lines');
% cls = reshape(cls, [numel(unique(rows)) numel(unique(cols))]);
cls_ori = zeros(size(sampleMask));
cls_ori(sampleMask) = cls;

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
layerImg = zeros(size(teImg));
for cInd=1:nCls
    layerImg_c = zeros(size(teImg));
    layerImg_c(repmat(cls_ori == cInd, [1 1 3])) = 1;
    layerImg = layerImg + bsxfun(@times, layerImg_c, reshape(lineCols(cInd, :), [1 1 3]));
end
h = imagesc(layerImg);
set(h, 'AlphaData', 0.7);
hold off;
title('Predicted regions');


end

