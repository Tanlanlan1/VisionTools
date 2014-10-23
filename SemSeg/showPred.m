function [o_hFig] = showPred( pred, params, trImg, trLabels, teImg, outFFmt )
%SHOWPRED Summary of this function goes here
%   Detailed explanation goes here

%% init

if nargin == 5
    outFFmt = [];
end

nCls = params.classifier.nCls - 1; % rm bg

% nCls = size(vals, 2) - 1;
% sampleMask = params.feat.sampleMask;
% cls_ori = zeros(size(sampleMask));
% cls_ori(sampleMask) = cls;

o_hFig = zeros(numel(pred), 1);
for cfInd=1:numel(pred)
    cls_ori = pred(cfInd).cls;

    hFig = 3000 + cfInd;
    %% show
    figure(hFig); clf;
    lineCols = colormap(lines);
    nMaxColor = size(lineCols, 1);
    subplot_tight(2, 2, 1);
    imshow(trImg);
    hold on;
    layerImg = zeros(size(trImg));
    clsColors = zeros(nCls, 3);
%     for cInd=1:nCls
%         layerImg_c = zeros(size(trImg));
%         layerImg_c(repmat(trLabels.cls == cInd, [1 1 3])) = 1;
%         layerImg = layerImg + bsxfun(@times, layerImg_c, reshape(lineCols(mod(cInd-1, nMaxColor)+1, :), [1 1 3]));
%         clsColors(cInd, :) = lineCols(mod(cInd-1, nMaxColor)+1, :);
%     end
    cInd = cfInd;
    layerImg_c = zeros(size(trImg));
    layerImg_c(repmat(trLabels.cls == cInd, [1 1 3])) = 1;
    layerImg = layerImg + bsxfun(@times, layerImg_c, reshape(lineCols(mod(cInd-1, nMaxColor)+1, :), [1 1 3]));
    clsColors(cInd, :) = lineCols(mod(cInd-1, nMaxColor)+1, :);
    
    h = imagesc(layerImg);
    set(h, 'AlphaData', 0.7);
    hold off;
    title('training regions');

    subplot_tight(2, 2, 2);
    imshow(teImg);
    hold on;
    layerImg = zeros(size(teImg));
    if params.classifier.binary %%FIXME: any good way to handle it? 
        cInd = 1;
        layerImg_c = zeros(size(teImg));
        layerImg_c(repmat(cls_ori == cInd, [1 1 3])) = 1;
        layerImg = layerImg + bsxfun(@times, layerImg_c, reshape(lineCols(mod(cInd-1, nMaxColor)+1, :), [1 1 3]));
    else
        for cInd=1:nCls
            layerImg_c = zeros(size(teImg));
            layerImg_c(repmat(cls_ori == cInd, [1 1 3])) = 1;
            layerImg = layerImg + bsxfun(@times, layerImg_c, reshape(clsColors(cInd, :), [1 1 3]));
        end
    end
    h = imagesc(layerImg);
    set(h, 'AlphaData', 0.7);
    hold off;
    title('Predicted regions');

    % show bbs
    subplot_tight(2, 2, 3);
    showbbs(teImg, pred(cfInd).bbs, inf);
    
    if ~isempty(outFFmt)
        saveas(hFig, outFFmt, 'png');
    end

    o_hFig(cfInd) = hFig;
end
end

