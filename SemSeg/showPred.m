function [o_hFig] = showPred( pred, params, trImgs, trLabels, teImgs, outFFmt )
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
pivotScaleInd = [teImgs(:).pivot];
if ~any(pivotScaleInd)
    assert(numel(teImgs) == 1)
    teImgs(1).img = imresize(teImgs(1).img, 1/teImgs(1).scale);
    teImgs(1).scale = 1;
    pivotScaleInd = 1;
end

o_hFig = [];
hFig = 3000;
for iInd1=1:size(teImgs, 1)
    for iInd2=1:size(teImgs, 2)
        teImg = teImgs(iInd1, iInd2, pivotScaleInd).img;
        for cfInd=1:numel(pred)
            %% find proper training example
            trImg = [];
            trLabel = [];
            for tiInd=1:numel(trImgs)
                l = trLabels(tiInd).cls(:, :,cfInd);
                if any(l(:))
                    trImg = trImgs(tiInd).img;
                    trLabel = l;
                    break;
                end
            end
            if isempty(trImg)
                % the object does not exist in given training images
                continue;
            end
            assert(~isempty(trImg));

            %% show an example training image
            hFig = hFig + 1;
            figure(hFig); clf;
            lineCols = colormap(lines);
            nMaxColor = size(lineCols, 1);
            subplot_tight(2, 2, 1);
            imshow(trImg);

            %% show the corresponding label
            hold on;
            layerImg = zeros(size(trImg));
            clsColors = zeros(nCls, 3);
            cInd = cfInd;
            layerImg_c = zeros(size(trImg));
            layerImg_c(logical(repmat(trLabel, [1 1 3]))) = 1;
            layerImg = layerImg + bsxfun(@times, layerImg_c, reshape(lineCols(mod(cInd-1, nMaxColor)+1, :), [1 1 3]));
            clsColors(cInd, :) = lineCols(mod(cInd-1, nMaxColor)+1, :);
            % show layer
            h = imagesc(layerImg);
            set(h, 'AlphaData', 0.7);
            hold off;
            title('training regions');
            %% show the test image and the corresponding predicted label
            cls_ori = pred(cfInd).cls;
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

            %% show bbs
            subplot_tight(2, 2, 3);
            showbbs(teImg, pred(cfInd).bbs, inf);

            %% save figures
            if ~isempty(outFFmt)
                saveas(hFig, outFFmt, 'png');
            end

            o_hFig = [o_hFig; hFig];
        end
    end
end
end

