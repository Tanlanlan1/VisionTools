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
    for iInd=1:numel(imgs_test)
        for sInd=1:numel(i_params.scales)
            imgs_test_scale(iInd, sInd).img = imresize(imgs_test(iInd).img, i_params.scales(sInd));
            imgs_test_scale(iInd, sInd).scale = i_params.scales(sInd);
            imgs_test_scale(iInd, sInd).pivot = i_params.scales(sInd) == 1;
        end
    end
    i_imgs = imgs_test_scale;
end

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
% LOFilterWH_half = (i_params.feat.LOFilterWH-1)/2; 
nData_approx = sum(prod(imgWHs, 1));

ixy = zeros(3, nData_approx);
% supInd = zeros(1, nData_approx);
supLabelSts = cell(nImgs, 1); % assumes only image with multiple scales
startInd = 1;
feats = [];
for iInd=1:nImgs

    % extract features
    [feat, tbParams] = GetDenseFeature(i_imgs(iInd), {'TextonBoostInt'}, i_params.feat);
    feats = [feats; feat];

    if i_params.supPix == 1
        % get superpixel
        if isfield(i_imgs(iInd), 'Superpixel')
            supLabelSts{iInd} = i_imgs(iInd).Superpixel;
        else
            [~, supLabelSts{iInd}] = GetSuperpixel(i_imgs(iInd).img, 'SLIC');
        end
        label_xymean = supLabelSts{iInd}.lblXYMean;

        % construct meta data
        ixy(:, startInd:startInd+size(label_xymean, 2)-1) = [iInd*ones(1, size(label_xymean, 2)); label_xymean];

        % update a pointer
        startInd = startInd+size(label_xymean, 2);
    else        
        % build sampleMask
        sampleMask = sampleMasks(iInd).mask;

        % construct meta data
        [rows, cols] = find(sampleMask);
        xy = [cols'; rows']; % be careful the order
        ixy(:, startInd:startInd+size(xy, 2)-1) = [iInd*ones(1, size(xy, 2)); xy];

        % update a pointer
        startInd = startInd+size(xy, 2);
    end
    
    
end
ixy(:, startInd:end) = [];


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
if JBParams.verbosity >= 1
    fprintf('* Running time PredSemSeg_mex: %s sec.\n', num2str(toc(mexTID)));
end

%% reshape responses
mexTID = tic;
dist_resh = reshapePredResponse_mex(size(i_imgs, 1), size(i_imgs, 2), ixy, imgWHs, dist, cell2mat(supLabelSts));
if JBParams.verbosity >= 1
    fprintf('* Running time reshapePredResponse_mex: %s sec.\n', num2str(toc(mexTID)));
end
pred = predLabel(i_params, i_imgs, dist_resh);

%% return
o_params = struct('feat', tbParams, 'classifier', JBParams);
o_feats = reshape(feats, size(i_imgs));
o_pred = pred;
if JBParams.verbosity >= 1
    fprintf('* The total running time of PredSemSeg.m: %s\n\n', num2str(toc(totTic)));
end

end


function [o_resp] = predLabel(i_params, i_imgs, i_dist_resh)

nmsOvRatio = 0.5;

assert(size(i_imgs, 1) == 1); %%FIXME: assume only one image with different scale images
iInd = 1;
refImgInd = find([i_imgs(iInd, :).pivot]);
% [~, refImgInd] = min(abs([i_imgs(iInd, :).scale] - 1));
refScale = i_imgs(refImgInd).scale;
imgWH_s1 = [size(i_imgs(iInd, refImgInd).img, 2); size(i_imgs(iInd, refImgInd).img, 1)];

nCls = size(i_dist_resh(1).resp, 3);
nClf = size(i_dist_resh(1).resp, 4);

pred = struct(...
    'dist', zeros(imgWH_s1(2), imgWH_s1(1), nCls), ...
    'cls', zeros(imgWH_s1(2), imgWH_s1(1)), ...
    'bbs', []);

for cfInd=1:nClf % for all classifiers

    dist_max_s = zeros(imgWH_s1(2), imgWH_s1(1), nCls);
    bbs = [];
    for iInd=1:size(i_imgs, 1)
        dist_s = zeros(imgWH_s1(2), imgWH_s1(1), size(i_imgs, 2), nCls);
        for sInd=1:size(i_imgs, 2)
            for cInd=1:nCls
                % set dist_S
                dist_s(:, :, sInd, cInd) = imresize(i_dist_resh(iInd, sInd).resp(:, :, cInd, cfInd), [size(dist_s, 1), size(dist_s, 2)]);

                % find bbs
                if i_params.classifier.binary == 1 && cInd == 1
                    curScale = i_params.scales(sInd);
                    [curBBs_rect, curScore] = GetBBs(i_params.mdlRects(cfInd, 3:4), i_dist_resh(iInd, sInd).resp(:, :, cInd, cfInd)); %%FIXME: return only one bb
                    curBBs_rect = curBBs_rect/curScale*refScale;
                    bbs = [bbs; curBBs_rect(1) curBBs_rect(2) curBBs_rect(1)+curBBs_rect(3)-1 curBBs_rect(2)+curBBs_rect(4)-1 curScore];
                end
            end
        end
%         dist_max_s(:, :, :) = squeeze(mean(dist_s, 3)); 
        dist_max_s(:, :, :) = squeeze(max(dist_s, [], 3));
    end
    % pixel: non-max suppression
    pred(cfInd).dist = dist_max_s;
    [~, pred(cfInd).cls] = max(dist_max_s, [], 3);
    % bbs: non-max suppression
    pred(cfInd).bbs = bbs(nms(bbs, nmsOvRatio), :);
end

%% return 
o_resp = pred;

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
maxResp = conv2(respMap, ones(bbWH(2), bbWH(1))./(bbWH(2)*bbWH(1)), 'valid');
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

