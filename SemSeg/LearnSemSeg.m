function [ o_mdl ] = LearnSemSeg( i_imgs, i_labels, i_params )
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

addpath(genpath('../Feature')); %%FIXME
addpath(genpath('../JointBoost')); %%FIXME

assert(isfield(i_imgs, 'img'));
assert(numel(i_imgs) == 1);

nImgs = numel(i_imgs);
%%FIXME: use same samplingRatio for texton and jointboost
samplingRatio = i_params.feat.samplingRatio;
%% extract image features (TextonBoost)
% imgs = cell(nImgs, 1);
% for iInd=1:nImgs
%     imgs{iInd} = i_imgs(iInd).img;
% end
imgs = i_imgs(1).img;
%%FIXME: struct array input/struct array output
% [feat, tbParams] = GetDenseFeature(imgs, {'TextonBoostInt'}, i_params.feat);
% [feat, tbParams] = GetDenseFeature(imgs, {'TextonBoost'}, i_params.feat); 

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
    
    % feature %%FIXME: tmp location!
    i_params.feat.sampleMask = sampleMask;
    [feat, tbParams] = GetDenseFeature(imgs, {'TextonBoostInt'}, i_params.feat); 
    
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
x_meta = struct('ixy', int32(ixy), 'intImgfeat', double(feat), 'tbParams', tbParams);

% run
JBParams = i_params.classifier;
JBParams.nData = numel(label);
if JBParams.verbosity >= 1
    fprintf('* Train %d data\n', JBParams.nData);
end
mdl = TrainJointBoost(@FeatCBFunc, label, JBParams, x_meta);
% mdl = TrainJointBoost(reshape(feat, [size(feat, 1)*size(feat, 2) size(feat, 3)]), label, JBParams);

% check input parameters
mdl = LearnSemSeq_mex(x_meta, int32(label), JBParams);

%% return
o_mdl = struct('JBMdl', mdl, 'TBParams', tbParams);

end

function [o_val] = FeatCBFunc(i_i, i_fInd, i_meta)
o_val = GetithTextonBoost( i_meta.intImgfeat, i_meta.ixy(:, i_i), i_fInd, i_meta.tbParams );
end

