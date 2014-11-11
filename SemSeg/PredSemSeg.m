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
totTic = tic;
% bulid a scale space
if ~isfield(i_imgs, 'scale')
    imgs_test_scale = struct('img', [], 'scale', [], 'pivot', []);
    imgs_test = i_imgs;
    for iInd1=1:size(imgs_test, 1)
        for iInd2=1:size(imgs_test, 2)
            for sInd=1:numel(i_params.scales)
                imgs_test_scale(iInd1, iInd2, sInd).img = imresize(imgs_test(iInd1, iInd2).img, i_params.scales(sInd));
                imgs_test_scale(iInd1, iInd2, sInd).scale = i_params.scales(sInd);
                imgs_test_scale(iInd1, iInd2, sInd).pivot = i_params.scales(sInd) == 1;
            end
        end
    end
    i_imgs = imgs_test_scale;
end
% precompute imgWHs
nImgs = numel(i_imgs); % linear indexing
imgWHs = zeros(2, nImgs);
for iInd=1:nImgs
    imgWHs(:, iInd) = [size(i_imgs(iInd).img, 2); size(i_imgs(iInd).img, 1)];
end

%% extract features 
tbParams = i_params.feat; % should not be changed
% extract Texture
[feat_texture, tbParams] = GetDenseFeature(i_imgs, {'Texture_LM'}, tbParams);
% extract integral images
% feats = arrayfun(@(x) GetDenseFeature(x, {'TextonBoostInt'}, i_params.feat), feat_texture);
feats = GetDenseFeature(feat_texture, {'TextonBoostInt'}, i_params.feat);

%% construct data structure
[ixy, supLabelSts] = sampleData(i_params, i_imgs);
x_meta = struct('ixy', struct('ixys_cls', ixy), 'intImgFeat', feats, 'TBParams', tbParams, 'imgWHs', imgWHs, 'supLabelSt', supLabelSts);

%% predict
JBParams = i_params.classifier;
JBParams.nData = size(ixy, 2);
if JBParams.verbosity >= 1
    fprintf('* Predict %d data\n', JBParams.nData);
end
[x_meta_mex, ~, JBParams_mex, ] = convParamsType(x_meta, [], JBParams);
mexTID = tic;
dist_resh = PredSemSeg_mex(x_meta_mex, i_mdls, JBParams_mex);
if JBParams.verbosity >= 1
    fprintf('* PredSemSeg_mex running time: %s sec.\n', num2str(toc(mexTID)));
end

% %% reshape responses
% mexTID = tic;
% dist_resh = reshapePredResponse_mex(nImgs, ixy, imgWHs, dist, supLabelSts);
% if JBParams.verbosity >= 1
%     fprintf('* reshapePredResponse_mex running time: %s sec.\n', num2str(toc(mexTID)));
% end

%% prediction results
mexTID = tic;
imgs = i_imgs;
dist_resh = reshape(dist_resh, size(imgs));
refImgInd = find([imgs(1, 1, :).pivot]);
if isempty(refImgInd)
    assert(numel(imgs) == 1);
    imgs(2).img = imresize(imgs(1).img, 1/imgs(1).scale);
    imgs(2).scale = 1;
    refImgInd = 2;
end
refScale = imgs(refImgInd).scale;
imgWH_s1 = [size(imgs(1, 1, refImgInd).img, 2); size(imgs(1, 1, refImgInd).img, 1)];
pred = predLabel(i_params, feats, imgWH_s1, refScale, dist_resh);
if JBParams.verbosity >= 1
    fprintf('* predLabel running time: %s sec.\n', num2str(toc(mexTID)));
end
%% return
o_params = struct('feat', tbParams, 'classifier', JBParams);
o_feats = reshape(feats, size(i_imgs));
o_pred = pred; 
if JBParams.verbosity >= 1
    fprintf('* The total running time of PredSemSeg.m: %s\n\n', num2str(toc(totTic)));
end

end

function [ixy, supLabelSts] = sampleData(i_params, i_imgs)

nImgs = numel(i_imgs);
ixy = cell(1, nImgs);
supLabelSts = cell(nImgs, 1); 
%% sample
parfor iInd=1:nImgs
    if i_params.supPix == 1
        % sparse sampling by using superpixel
        if isfield(i_imgs(iInd), 'Superpixel')
            supLabelSts{iInd} = i_imgs(iInd).Superpixel;
        else
            [~, supLabelSts{iInd}] = GetSuperpixel(i_imgs(iInd).img, 'SLIC');
        end
        label_xymean = supLabelSts{iInd}.lblXYMean;
        ixy{iInd} = [iInd*ones(1, size(label_xymean, 2)); label_xymean];
    else        
        % dense sampling
        sampleMask = true(size(i_imgs(iInd).img, 1), size(i_imgs(iInd).img, 2));
        [rows, cols] = find(sampleMask);
        xy = [cols'; rows']; % be careful the order
        ixy{iInd} = [iInd*ones(1, size(xy, 2)); xy];
    end
end

%% return
ixy = cell2mat(ixy);
supLabelSts = cell2mat(supLabelSts);
end

function [o_resp] = predLabel(i_params, i_imgs, i_pivotImgWH, i_pivotScale, i_dist_resh)

nmsOvRatio = 0.5;

refScale = i_pivotScale;
imgWH_s1 = i_pivotImgWH;

nImg1 = size(i_dist_resh, 1);
nImg2 = size(i_dist_resh, 2);
nScales = size(i_dist_resh, 3);
nCls = size(i_dist_resh(1).resp, 3);
nClf = size(i_dist_resh(1).resp, 4);

pred = struct(...
    'dist', zeros(imgWH_s1(2), imgWH_s1(1), nCls), ...
    'cls', zeros(imgWH_s1(2), imgWH_s1(1)), ...
    'bbs', []);

for cfInd=1:nClf % for all classifiers
    dist_max_s = zeros(imgWH_s1(2), imgWH_s1(1), nCls);
    bbs = [];
    for iInd1=1:nImg1
        for iInd2=1:nImg2
            dist_s = zeros(imgWH_s1(2), imgWH_s1(1), nScales, nCls);
            iInd = sub2ind([nImg1, nImg2], iInd1, iInd2);
            for sInd=1:nScales
                for cInd=1:nCls
                    % set dist_S
                    dist_s(:, :, sInd, cInd) = imresize(i_dist_resh(iInd1, iInd2, sInd).resp(:, :, cInd, cfInd), [size(dist_s, 1), size(dist_s, 2)]);

                    % find bbs
                    if i_params.classifier.binary == 1 && cInd == 1 && ~isempty(i_params.mdlRects)
                        curScale = i_imgs(iInd1, iInd2, sInd).scale;
%                         curScale = i_params.scales(sInd);
                        curPoly = i_params.mdlRects(cfInd).iInd(iInd).poly;
                        curMdlMask = poly2mask(curPoly(1, :), curPoly(2, :), round(max(curPoly(2, :))), round(max(curPoly(1, :))));
                        curMdlMask(:, ~any(curMdlMask, 1)) = [];
                        curMdlMask(~any(curMdlMask, 2), :) = [];
                        [curBBs_rect, curScore] = GetBBs(curMdlMask, i_dist_resh(iInd1, iInd2, sInd).resp(:, :, cInd, cfInd));
%                         [curBBs_rect, curScore] = GetBBs(i_params.mdlRects(cfInd, 3:4), i_dist_resh(iInd1, iInd2, sInd).resp(:, :, cInd, cfInd));
                        curBBs_rect = curBBs_rect/curScale*refScale;
                        bbs = [bbs; curBBs_rect(1) curBBs_rect(2) curBBs_rect(1)+curBBs_rect(3)-1 curBBs_rect(2)+curBBs_rect(4)-1 curScore];
                    end
                end
            end
    %         dist_max_s(:, :, :) = squeeze(mean(dist_s, 3)); 
            dist_max_s(:, :, :) = squeeze(max(dist_s, [], 3));
        end
    end
    
    
%     for iInd=1:size(i_imgs, 1)
%         dist_s = zeros(imgWH_s1(2), imgWH_s1(1), size(i_imgs, 2), nCls);
%         for sInd=1:size(i_imgs, 2)
%             for cInd=1:nCls
%                 % set dist_S
%                 dist_s(:, :, sInd, cInd) = imresize(i_dist_resh(iInd, sInd).resp(:, :, cInd, cfInd), [size(dist_s, 1), size(dist_s, 2)]);
% 
%                 % find bbs
%                 if i_params.classifier.binary == 1 && cInd == 1
%                     curScale = i_params.scales(sInd);
%                     [curBBs_rect, curScore] = GetBBs(i_params.mdlRects(cfInd, 3:4), i_dist_resh(iInd, sInd).resp(:, :, cInd, cfInd)); %%FIXME: return only one bb
%                     curBBs_rect = curBBs_rect/curScale*refScale;
%                     bbs = [bbs; curBBs_rect(1) curBBs_rect(2) curBBs_rect(1)+curBBs_rect(3)-1 curBBs_rect(2)+curBBs_rect(4)-1 curScore];
%                 end
%             end
%         end
% %         dist_max_s(:, :, :) = squeeze(mean(dist_s, 3)); 
%         dist_max_s(:, :, :) = squeeze(max(dist_s, [], 3));
%     end
    % pixel: non-max suppression
    pred(cfInd).dist = dist_max_s;
    [~, pred(cfInd).cls] = max(dist_max_s, [], 3);
    % bbs: non-max suppression
    pred(cfInd).bbs = bbs(nms(bbs, nmsOvRatio), :);
end

%% return 
o_resp = pred;

end


function [o_BBs_rect, o_score] = GetBBs(i_mdlMask, i_respMap)

i_mdlWH = [size(i_mdlMask, 2); size(i_mdlMask, 1)];
bbWH = round(i_mdlWH);
respMap = i_respMap;
% % show the response map
% if verbosity>=2
%     figure(43125); imagesc(respMap); axis image;
% end

% find max
respMap = padarray(respMap, max(0, [bbWH(2)-size(respMap, 1) bbWH(1)-size(respMap, 2)]), 0, 'pre'); 
mask_norm = i_mdlMask./sum(i_mdlMask(:));
maxResp = conv2(respMap, mask_norm, 'valid');
% % show the max response map
% if verbosity>=2
%     figure(56433); imagesc(maxResp); axis image;
% end

% return bb  %%FIXME: return only one bb
[C, y_maxs] = max(maxResp, [], 1);
[maxVal, xc] = max(C, [], 2);
yc = y_maxs(xc);

assert(~isempty(xc));
% o_BBs_rect = [xc-round(bbWH(1)/2) yc-round(bbWH(2)/2) bbWH(1) bbWH(2)];
o_BBs_rect = [xc yc bbWH(1) bbWH(2)];
o_score = maxVal;

end


% 
% function [o_BBs_rect, o_score] = GetBBs(i_mdlWH, i_respMap)
% 
% bbWH = round(i_mdlWH);
% respMap = i_respMap;
% % % show the response map
% % if verbosity>=2
% %     figure(43125); imagesc(respMap); axis image;
% % end
% 
% % find max
% respMap = padarray(respMap, max(0, [bbWH(2)-size(respMap, 1) bbWH(1)-size(respMap, 2)]), 0, 'pre'); 
% maxResp = conv2(respMap, ones(bbWH(2), bbWH(1))./(bbWH(2)*bbWH(1)), 'valid');
% % % show the max response map
% % if verbosity>=2
% %     figure(56433); imagesc(maxResp); axis image;
% % end
% 
% % return bb  %%FIXME: return only one bb
% [C, y_maxs] = max(maxResp, [], 1);
% [maxVal, xc] = max(C, [], 2);
% yc = y_maxs(xc);
% 
% assert(~isempty(xc));
% % o_BBs_rect = [xc-round(bbWH(1)/2) yc-round(bbWH(2)/2) bbWH(1) bbWH(2)];
% o_BBs_rect = [xc yc bbWH(1) bbWH(2)];
% o_score = maxVal;
% 
% end

