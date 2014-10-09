function [ o_cls, o_dist, o_params, o_feats ] = PredSemSeg( i_imgs, i_mdls, i_params )
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
nImgs = numel(i_imgs);

%% extract features and predict
LOFilterWH_half = (i_params.feat.LOFilterWH-1)/2; 
imgWH = [size(i_imgs(1).img, 2); size(i_imgs(1).img, 1)];
nData_approx = round(nImgs*imgWH(1)*imgWH(2)*1); %%FIXME: assume same sized images

ixy = zeros(3, nData_approx);
startInd = 1;
feats = [];
for iInd=1:nImgs
    % build sampleMask
    sampleMask = true(imgWH(2), imgWH(1));
    
    % falsify boundaries
    sampleMask(1:LOFilterWH_half(2), :) = false;
    sampleMask(imgWH(2)-LOFilterWH_half(2):end, :) = false;
    sampleMask(:, 1:LOFilterWH_half(1)) = false;
    sampleMask(:, imgWH(1)-LOFilterWH_half(1):end, :) = false;
    
    % extract features
    i_params.feat.sampleMask = sampleMask;
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
x_meta = struct('ixy', ixy, 'intImgFeat', feat, 'TBParams', tbParams);

% predict
JBParams = i_params.classifier;
JBParams.nData = size(ixy, 2);
if JBParams.verbosity >= 1
    fprintf('* Predict %d data\n', JBParams.nData);
end
[x_meta_mex, JBParams_mex] = convType(x_meta, JBParams);
mexTID = tic;
dist = PredSemSeg_mex(x_meta_mex, i_mdls, JBParams_mex);
fprintf('* Running time PredSemSeg_mex: %s sec.\n', num2str(toc(mexTID)));

%% return
o_params = struct('feat', tbParams, 'classifier', JBParams);
o_feats = feats;
[~, cls] = max(dist, [], 2);
% o_dist = dist;

assert(nImgs == 1);
sampleMask = o_params.feat.sampleMask;

o_cls = zeros(size(sampleMask));
o_cls(sampleMask) = cls;

o_dist = repmat(zeros(size(sampleMask)), [1 1 size(dist, 2)]);
for cInd=1:size(dist, 2)
    dist_tmp = zeros(size(sampleMask));
    dist_tmp(sampleMask) = dist(:, cInd);
    o_dist(:, :, cInd) = dist_tmp;
end
end

function [x_meta_mex, JBParams_mex] = convType(x_meta, JBParams)
x_meta_mex = x_meta;
% x_meta.ixy
x_meta_mex.ixy = int32(x_meta_mex.ixy);
% x_meta.intImgfeat
for iInd=1:numel(x_meta_mex.intImgFeat)
    x_meta_mex.intImgFeat(iInd).TextonIntImg = double(x_meta_mex.intImgFeat(iInd).TextonIntImg);
end
% x_meta.tbParams
x_meta_mex.TBParams.LOFilterWH = int32(x_meta_mex.TBParams.LOFilterWH);
x_meta_mex.TBParams.nTexton = int32(x_meta_mex.TBParams.nTexton);
x_meta_mex.TBParams.parts = int32(x_meta_mex.TBParams.parts);
% JBParams
JBParams_mex = JBParams;
JBParams_mex.nWeakLearner = int32(JBParams_mex.nWeakLearner);
JBParams_mex.featDim = int32(JBParams_mex.featDim);
JBParams_mex.nData = int32(JBParams_mex.nData);
JBParams_mex.nCls = int32(JBParams_mex.nCls);
JBParams_mex.featSelRatio = double(JBParams_mex.featSelRatio);
JBParams_mex.featValRange = double(JBParams_mex.featValRange(:));
JBParams_mex.verbosity = int32(JBParams_mex.verbosity);

end