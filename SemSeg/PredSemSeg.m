function [ o_pred, o_params, o_feats ] = PredSemSeg( i_imgs, i_mdls, i_params )
% 
%   Learn a semantic segmentation model
%   
% ----------
%   Input: 
% 
%       i_imgs                      a structure array of images
%           i_imgs.filename         an image filename
%           i_imgs.img              an image array
% 
%       i_labels                    a structure array of labels
%           i_labels.cls            a semantic class label
%           i_labels.depth          a depth
% 
%       i_params                    parameters
%           i_params.clsList        a list of class IDs
%           i_params.feat           feature extraction paramters
%           i_params.classifier     classifier parameters
%           
% ----------
%   Output:
% 
%       o_mdl                   a learned semantic segmentation model
% 
% ----------
%   Dependency:
%   
%
% ----------
% Written by Sangdon Park (sangdonp@cis.upenn.edu), 2014.
% All rights reserved.
%

%% init
if ~isfield(i_params, 'scales')
    i_params.scales = 1;
end
if ~isfield(i_params, 'mdlRects')
    i_params.mdlRects = [];
end


% bulid a scale space
imgs_test_scale = struct('img', [], 'scale', []);
imgs_test = i_imgs;
for iInd=1:numel(imgs_test)
    for sInd=1:numel(i_params.scales)
        imgs_test_scale(iInd, sInd).img = imresize(imgs_test(iInd).img, i_params.scales(sInd));
        imgs_test_scale(iInd, sInd).scale = i_params.scales(sInd);
    end
end
i_imgs = imgs_test_scale;

%%
assert(size(i_imgs, 1) == 1); %%FIXME: assume only one image with different scale images
nImgs = numel(i_imgs);
imgWHs = zeros(2, nImgs);
sampleMasks = struct('mask', []);
for iInd=1:nImgs
    imgWHs(:, iInd) = [size(i_imgs(iInd).img, 2); size(i_imgs(iInd).img, 1)];
    sampleMasks(iInd).mask = true(imgWHs(2, iInd), imgWHs(1, iInd));
end


%% extract features and predict
LOFilterWH_half = (i_params.feat.LOFilterWH-1)/2; 
nData_approx = sum(prod(imgWHs, 1));

ixy = zeros(3, nData_approx);
% supInd = zeros(1, nData_approx);
supLbl = cell(size(i_imgs)); % assumes only image with multiple scales
startInd = 1;
feats = [];
for iInd=1:nImgs
    
    % extract features
    [feat, tbParams] = GetDenseFeature(i_imgs(iInd), {'TextonBoostInt'}, i_params.feat);
    feats = [feats; feat];

%     % build sampleMask
%     sampleMask = sampleMasks(iInd).mask;
% 
%     % construct meta data
%     [rows, cols] = find(sampleMask);
%     xy = [cols'; rows']; % be careful the order
%     ixy(:, startInd:startInd+size(xy, 2)-1) = [iInd*ones(1, size(xy, 2)); xy];
%     
%     % update a pointer
%     startInd = startInd+size(xy, 2);
    
    % get superpixel
    [~, label_sub] = GetSuperpixel(i_imgs(iInd).img, 'SLIC');
    
    % construct meta data
    xy = cell2mat(cellfun(@(x) round(mean(x, 2)), label_sub, 'UniformOutput', false));
    ixy(:, startInd:startInd+size(xy, 2)-1) = [iInd*ones(1, size(xy, 2)); xy];
%     supInd(1, startInd:startInd+size(xy, 2)-1) = 1:size(xy, 2);
    supLbl{iInd} = label_sub;
    
    % update a pointer
    startInd = startInd+size(xy, 2);
end
ixy(:, startInd:end) = [];
% supInd(:, startInd:end) = [];

x_meta = struct('ixy', ixy, 'intImgFeat', feats, 'TBParams', tbParams);

% predict
JBParams = i_params.classifier;
JBParams.nData = size(ixy, 2);
if JBParams.verbosity >= 1
    fprintf('* Predict %d data\n', JBParams.nData);
end
[x_meta_mex, ~, JBParams_mex] = convParamsType(x_meta, [], JBParams);
mexTID = tic;
dist = PredSemSeg_mex(x_meta_mex, i_mdls, JBParams_mex);
fprintf('* Running time PredSemSeg_mex: %s sec.\n', num2str(toc(mexTID)));

%% max_s
assert(size(i_imgs, 1) == 1); %%FIXME: assume only one image with different scale images
iInd = 1;
[~, refImgInd] = min(abs([i_imgs(iInd, :).scale] - 1));
refScale = i_imgs(refImgInd).scale;
imgWH_s1 = [size(i_imgs(iInd, refImgInd).img, 2); size(i_imgs(iInd, refImgInd).img, 1)];

pred = struct('dist', [], 'cls', [], 'bbs', []);
for cfInd=1:size(dist, 3) % for all classifiers

    dist_max_s = zeros(imgWH_s1(2), imgWH_s1(1), size(dist, 2));
    bbs = [];
    for iInd=1:size(i_imgs, 1)
        dist_s = zeros(imgWH_s1(2), imgWH_s1(1), size(i_imgs, 2), size(dist, 2));
        for sInd=1:size(i_imgs, 2)
            curScale = i_params.scales(sInd);
            for cInd=1:size(dist, 2)
                iInd_lin = sub2ind(size(i_imgs), iInd, sInd);
                curDist = dist(ixy(1, :) == iInd_lin, cInd, cfInd);

                % superpixel-wise response map to pixel-wise one. assume
                % vector index is same as superpixel id
                supSubInd = supLbl{iInd, sInd};
                assert(numel(supSubInd) == size(curDist, 1));
                dist_tmp = zeros(size(sampleMasks(iInd, sInd).mask));
                for supInd=1:size(curDist, 1)
                    dist_tmp(sub2ind(size(dist_tmp), supSubInd{supInd}(2, :), supSubInd{supInd}(1, :))) = curDist(supInd);
                end
                
                % set dist_S
                dist_s(:, :, sInd, cInd) = imresize(dist_tmp, [size(dist_s, 1), size(dist_s, 2)]);
                % find bbs
                if i_params.classifier.binary == 1 && cInd == 1
                    [curBBs_rect, curScore] = GetBBs(i_params.mdlRects(cfInd, 3:4), dist_tmp); %%FIXME: return only one bb
                    curBBs_rect = curBBs_rect/curScale*refScale;
                    bbs = [bbs; curBBs_rect(1) curBBs_rect(2) curBBs_rect(1)+curBBs_rect(3)-1 curBBs_rect(2)+curBBs_rect(4)-1 curScore];
                end
            end
        end
%         dist_max_s(:, :, :) = squeeze(max(dist_s, [], 3));
        dist_max_s(:, :, :) = squeeze(mean(dist_s, 3));
    end
    % pixel: non-max suppression
    pred(cfInd).dist = dist_max_s;
    [~, pred(cfInd).cls] = max(dist_max_s, [], 3);
    % bbs: non-max suppression
    pred(cfInd).bbs = bbs(nms(bbs, 0.5), :);
end


% %% max_s
% assert(size(i_imgs, 1) == 1); %%FIXME: assume only one image with different scale images
% iInd = 1;
% [~, refImgInd] = min(abs([i_imgs(iInd, :).scale] - 1));
% refScale = i_imgs(refImgInd).scale;
% imgWH_s1 = [size(i_imgs(iInd, refImgInd).img, 2); size(i_imgs(iInd, refImgInd).img, 1)];
% 
% pred = struct('dist', [], 'cls', [], 'bbs', []);
% for cfInd=1:size(dist, 3)
% 
%     dist_max_s = zeros(imgWH_s1(2), imgWH_s1(1), size(dist, 2));
%     bbs = [];
%     for iInd=1:size(i_imgs, 1)
%         dist_s = zeros(imgWH_s1(2), imgWH_s1(1), size(i_imgs, 2), size(dist, 2));
%         for sInd=1:size(i_imgs, 2)
%             curScale = i_params.scales(sInd);
%             for cInd=1:size(dist, 2)
%                 iInd_lin = sub2ind(size(i_imgs), iInd, sInd);
%                 sampleMask = sampleMasks(iInd, sInd).mask;
%                 curDist = dist(ixy(1, :) == iInd_lin, :, cfInd);
% 
%                 % current scale response map
%                 dist_tmp = zeros(size(sampleMask));
%                 dist_tmp(sampleMask) = curDist(:, cInd);
%                 % set dist_S
%                 dist_s(:, :, sInd, cInd) = imresize(dist_tmp, [size(dist_s, 1), size(dist_s, 2)]);
%                 % find bbs
%                 if i_params.classifier.binary == 1 && cInd == 1
%                     [curBBs_rect, curScore] = GetBBs(i_params.mdlRects(cfInd, 3:4), dist_tmp); %%FIXME: return only one bb
%                     curBBs_rect = curBBs_rect/curScale*refScale;
%                     bbs = [bbs; curBBs_rect(1) curBBs_rect(2) curBBs_rect(1)+curBBs_rect(3)-1 curBBs_rect(2)+curBBs_rect(4)-1 curScore];
%                 end
%             end
%         end
% %         dist_max_s(:, :, :) = squeeze(max(dist_s, [], 3));
%         dist_max_s(:, :, :) = squeeze(mean(dist_s, 3));
%     end
%     % pixel: non-max suppression
%     pred(cfInd).dist = dist_max_s;
%     [~, pred(cfInd).cls] = max(dist_max_s, [], 3);
%     % bbs: non-max suppression
%     pred(cfInd).bbs = bbs(nms(bbs, 0.5), :);
% end


%% return
o_params = struct('feat', tbParams, 'classifier', JBParams);
o_feats = feats;
o_pred = pred;

end

function [o_BBs_rect, o_score] = GetBBs(i_mdlWH, i_respMap)

bbWH = round(i_mdlWH);
respMap = i_respMap;
% % show the response map
% if verbosity>=2
%     figure(43125); imagesc(respMap); axis image;
% end

% find max
respMap = padarray(respMap, max(0, [bbWH(2)-size(respMap, 1) bbWH(1)-size(respMap, 2)]), 0, 'pre'); 
maxResp = conv2(respMap, fspecial('average', [bbWH(2) bbWH(1)]), 'valid');
% % show the max response map
% if verbosity>=2
%     figure(56433); imagesc(maxResp); axis image;
% end

% return bb
[C, y_maxs] = max(maxResp, [], 1);
[maxVal, xc] = max(C, [], 2);
yc = y_maxs(xc);

assert(~isempty(xc));
% o_BBs_rect = [xc-round(bbWH(1)/2) yc-round(bbWH(2)/2) bbWH(1) bbWH(2)];
o_BBs_rect = [xc yc bbWH(1) bbWH(2)];
o_score = maxVal;

end