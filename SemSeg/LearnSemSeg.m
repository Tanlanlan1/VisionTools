function [ o_mdl, o_params ] = LearnSemSeg( i_imgs, i_labels, i_params )
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
%       o_mdl                       a learned semantic segmentation model
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

addpath(genpath('../Feature')); %%FIXME
addpath(genpath('../JointBoost')); %%FIXME

assert(isfield(i_imgs, 'img'));
assert(numel(i_imgs) == 1);

nImgs = numel(i_imgs);
%%FIXME: use same samplingRatio for texton and jointboost
samplingRatio = i_params.feat.samplingRatio;

%% learn a classifier (JointBoost)

% buid meta data for FeatCBFunc and labels
LOFilterWH_half = (i_params.feat.LOFilterWH-1)/2; 
imgWH = [size(i_imgs(1).img, 2); size(i_imgs(1).img, 1)];
nData_approx = round(nImgs*imgWH(1)*imgWH(2)*samplingRatio); %%FIXME: assume same sized images
step = round(1/samplingRatio);

ixy = zeros(3, nData_approx);
label = zeros(nData_approx, 1);
startInd = 1;
for iInd=1:nImgs
    % build sampleMask
    sampleMask = false(imgWH(2), imgWH(1));
    [rows, cols] = meshgrid(1:step:imgWH(2), 1:step:imgWH(1));
    sampleMask(rows, cols) = true;
    % falsify boundaries
    sampleMask(1:LOFilterWH_half(2), :) = false;
    sampleMask(imgWH(2)-LOFilterWH_half(2):end, :) = false;
    sampleMask(:, 1:LOFilterWH_half(1)) = false;
    sampleMask(:, imgWH(1)-LOFilterWH_half(1):end, :) = false;
    
    % extract feature (TextonBoost)
    i_params.feat.sampleMask = sampleMask;
    [feat, tbParams] = GetDenseFeature(i_imgs(iInd), {'TextonBoostInt'}, i_params.feat); 
    
    % construct meta data
    [rows, cols] = find(sampleMask);
    xy = [cols'; rows']; % be careful the order
    ixy(:, startInd:startInd+size(xy, 2)-1) = [iInd*ones(1, size(xy, 2)); xy];
    
    % construct label
    linInd = sub2ind(size(i_labels(iInd).cls), xy(2, :), xy(1, :));
    curLabel = i_labels(iInd).cls(linInd);
    label(startInd:startInd+size(xy, 2)-1) = curLabel;
    
    % update a pointer
    startInd = startInd+size(xy, 2);
end
ixy(:, startInd:end) = [];
label(startInd:end) = [];
x_meta = struct('ixy', ixy, 'intImgFeat', feat, 'TBParams', tbParams);

% run
JBParams = i_params.classifier;
JBParams.nData = numel(label);
if JBParams.verbosity >= 1
    fprintf('* Train %d data\n', JBParams.nData);
end
% check input parameters
[x_meta_mex, label_mex, JBParams_mex] = convType(x_meta, label, JBParams);
if JBParams.verbosity >= 1
    mexTID = tic;
end
mdls = LearnSemSeg_mex(x_meta_mex, label_mex, JBParams_mex);
if JBParams.verbosity >= 1
    fprintf('* Running time LearnSemSeg_mex: %s sec.\n', num2str(toc(mexTID)));
end

%% return
o_mdl = mdls; 
o_params = struct('feat', tbParams, 'classifier', JBParams);

end

function [x_meta_mex, label_mex, JBParams_mex] = convType(x_meta, label, JBParams)
x_meta_mex = x_meta;
% x_meta.ixy
x_meta_mex.ixy = int32(x_meta_mex.ixy);
% x_meta.intImgfeat
for iInd=1:numel(x_meta_mex.intImgFeat)
    x_meta_mex.intImgFeat(iInd).feat = double(x_meta_mex.intImgFeat(iInd).feat);
end
% x_meta.tbParams
x_meta_mex.TBParams.LOFilterWH = int32(x_meta_mex.TBParams.LOFilterWH);
x_meta_mex.TBParams.nTexton = int32(x_meta_mex.TBParams.nTexton);
x_meta_mex.TBParams.parts = int32(x_meta_mex.TBParams.parts);
% label
label_mex = int32(label(:));
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
