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
startInd = 1;
feats = [];
for iInd=1:nImgs
    % build sampleMask
    sampleMask = sampleMasks(iInd).mask;

    
%     % falsify boundaries
%     sampleMask(1:LOFilterWH_half(2), :) = false;
%     sampleMask(imgWH(2)-LOFilterWH_half(2):end, :) = false;
%     sampleMask(:, 1:LOFilterWH_half(1)) = false;
%     sampleMask(:, imgWH(1)-LOFilterWH_half(1):end, :) = false;
    
    % extract features
    [feat, tbParams] = GetDenseFeature(i_imgs(iInd), {'TextonBoostInt'}, i_params.feat);
    feats = [feats; feat];
    
    % construct meta data
    [rows, cols] = find(sampleMask);
    xy = [cols'; rows']; % be careful the order
    ixy(:, startInd:startInd+size(xy, 2)-1) = [iInd*ones(1, size(xy, 2)); xy];
    
    % update a pointer
    startInd = startInd+size(xy, 2);
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
fprintf('* Running time PredSemSeg_mex: %s sec.\n', num2str(toc(mexTID)));

%% max_s
assert(size(i_imgs, 1) == 1); %%FIXME: assume only one image with different scale images
iInd = 1;
[~, refImgInd] = min(abs([i_imgs(iInd, :).scale] - 1));
imgWH_s1 = [size(i_imgs(iInd, refImgInd).img, 2); size(i_imgs(iInd, refImgInd).img, 1)];

pred = struct('dist', [], 'cls', []);
for cfInd=1:size(dist, 3)

    dist_max_s = zeros(imgWH_s1(2), imgWH_s1(1), size(dist, 2));
    for iInd=1:size(i_imgs, 1)
%         refImgInd = [i_imgs(iInd, :).scale] == 1;
%         assert(~isempty(refImgInd));
%         imgWH_s1 = [size(i_imgs(iInd, refImgInd).img, 2); size(i_imgs(iInd, refImgInd).img, 1)];
        dist_s = zeros(imgWH_s1(2), imgWH_s1(1), size(i_imgs, 2), size(dist, 2));
        for sInd=1:size(i_imgs, 2)
%             curScale = i_imgs(iInd, sInd).scale;
            for cInd=1:size(dist, 2)
                iInd_lin = sub2ind(size(i_imgs), iInd, sInd);
                sampleMask = sampleMasks(iInd, sInd).mask;
                curDist = dist(ixy(1, :) == iInd_lin, :, cfInd);

                dist_tmp = zeros(size(sampleMask));
                dist_tmp(sampleMask) = curDist(:, cInd);
                %dist_s(:, :, sInd, cInd) = imresize(dist_tmp, 1/curScale);
                dist_s(:, :, sInd, cInd) = imresize(dist_tmp, [size(dist_s, 1), size(dist_s, 2)]);
            end
        end
        dist_max_s(:, :, :) = squeeze(max(dist_s, [], 3));
    end
    pred(cfInd).dist = dist_max_s;
    [~, pred(cfInd).cls] = max(dist_max_s, [], 3);
    
%     if i_params.classifier.binary == 1
%         pred(cfInd).cls = dist_max_s(:, :, 1)>0 + 2*(dist_max_s(:, :, 1)<=0);
%     else
%         [~, pred(cfInd).cls] = max(dist_max_s, [], 3); %%FIXME: how can we handle background??
%     end
    
end

% o_dist = repmat(zeros(size(sampleMask)), [1 1 size(dist, 2)]);
% for cInd=1:size(dist, 2)
%     dist_tmp = zeros(size(sampleMask));
%     dist_tmp(sampleMask) = dist(:, cInd);
%     o_dist(:, :, cInd) = dist_tmp;
% end


%% return
o_params = struct('feat', tbParams, 'classifier', JBParams);
o_feats = feats;
o_pred = pred;
% o_dist = dist_max_s;
% [~, o_cls] = max(o_dist, [], 3);

end

% function [x_meta_mex, JBParams_mex] = convType(x_meta, JBParams)
% x_meta_mex = x_meta;
% % x_meta.ixy
% x_meta_mex.ixy = int32(x_meta_mex.ixy);
% % x_meta.intImgfeat
% for iInd=1:numel(x_meta_mex.intImgFeat)
%     x_meta_mex.intImgFeat(iInd).TextonIntImg = double(x_meta_mex.intImgFeat(iInd).TextonIntImg);
% end
% % x_meta.tbParams
% x_meta_mex.TBParams.LOFilterWH = int32(x_meta_mex.TBParams.LOFilterWH);
% x_meta_mex.TBParams.nTexton = int32(x_meta_mex.TBParams.nTexton);
% x_meta_mex.TBParams.parts = int32(x_meta_mex.TBParams.parts);
% % JBParams
% JBParams_mex = JBParams;
% JBParams_mex.nWeakLearner = int32(JBParams_mex.nWeakLearner);
% JBParams_mex.featDim = int32(JBParams_mex.featDim);
% JBParams_mex.nData = int32(JBParams_mex.nData);
% JBParams_mex.nCls = int32(JBParams_mex.nCls);
% JBParams_mex.featSelRatio = double(JBParams_mex.featSelRatio);
% JBParams_mex.featValRange = double(JBParams_mex.featValRange(:));
% JBParams_mex.verbosity = int32(JBParams_mex.verbosity);
% 
% end