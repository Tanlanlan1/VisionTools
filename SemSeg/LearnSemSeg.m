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
%           i_params.pad            a boolean flag (true: pad images)
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

if ~isfield(i_params, 'pad')
    i_params.pad = false;
end
if ~isfield(i_params, 'nPerClsSample')
    i_params.nPerClsSample = 100;
end


nImgs = numel(i_imgs);
samplingRatio = i_params.feat.samplingRatio; %%FIXME: mask? saplingratio? duplicated
nPerClsSample = i_params.nPerClsSample;
tbParams = i_params.feat;
nCls = i_params.classifier.nCls;

%% learn a classifier (JointBoost)

% buid meta data for FeatCBFunc and labels
LOFilterWH_half = (i_params.feat.LOFilterWH-1)/2; 
% nData_approx = round(nImgs*size(i_imgs(1).img, 2)*size(i_imgs(1).img, 1)*samplingRatio); %%FIXME: assume same sized images
nData_approx = round(nImgs*size(i_imgs(1).img, 2)*size(i_imgs(1).img, 1)); %%FIXME: assume same sized images
% step = round(1/samplingRatio);

% extract Texton
[~, tbParams] = GetDenseFeature(i_imgs, {'Texton'}, tbParams); %%FIXME: sampling points are different with JointBoost

ixy = zeros(3, nData_approx);
label = zeros(nData_approx, 1);
startInd = 1;
for iInd=1:nImgs
    % pad
    if i_params.pad
        curImg = struct('img', padarray(i_imgs(iInd).img, [LOFilterWH_half(2) LOFilterWH_half(1) 0], 'symmetric', 'both'));
        curLabel = struct('cls', padarray(i_labels(iInd).cls, [LOFilterWH_half(2) LOFilterWH_half(1) 0], 'symmetric', 'both'));
    else
        curImg = struct('img', i_imgs(iInd).img);
        curLabel = struct('cls', i_labels(iInd).cls);
    end
    imgWH = [size(curImg.img, 2); size(curImg.img, 1)];
    
    % extract feature (TextonBoost) 
    [feat, tbParams] = GetDenseFeature(curImg, {'TextonBoostInt'}, tbParams); 
    
    % build sampleMask %%FIXME: should be balanced across images also...
    % falsify boundaries
    curLabel.cls(1:LOFilterWH_half(2), :) = nan;
    curLabel.cls(imgWH(2)-LOFilterWH_half(2):end, :) = nan;
    curLabel.cls(:, 1:LOFilterWH_half(1)) = nan;
    curLabel.cls(:, imgWH(1)-LOFilterWH_half(1):end, :) = nan;
    
    sampleMask_jb = zeros(imgWH(2), imgWH(1)); 
    for cInd=0:nCls % include bg=0
        [rows, cols] = find(curLabel.cls == cInd);
        if isempty(rows)
            continue;
        end
        
        rndInd = randi(numel(rows), [nPerClsSample, 1]);
        sampleMask_jb(rows(rndInd), cols(rndInd)) = sampleMask_jb(rows(rndInd), cols(rndInd)) + 1;
    end
        
    xys = [];
    while true     
        [rows, cols] = find(sampleMask_jb>=1);
        if isempty(rows)
            break;
        end
        sampleMask_jb(sampleMask_jb>=1) = sampleMask_jb(sampleMask_jb>=1) - 1;
        
        xy = [cols'; rows']; % be careful the order
        xys = [xys xy];    
    end
    ixy(:, startInd:startInd+size(xys, 2)-1) = [iInd*ones(1, size(xys, 2)); xys];
    
    % construct label
    linInd = sub2ind(size(curLabel.cls), xys(2, :), xys(1, :));
    label(startInd:startInd+size(xys, 2)-1) = curLabel.cls(linInd);
    
    % update a pointer
    startInd = startInd+size(xys, 2);
    
    
%     % build sampleMask
%     sampleMask_jb = false(imgWH(2), imgWH(1));
%     [rows, cols] = meshgrid(1:step:imgWH(2), 1:step:imgWH(1));
%     sampleMask_jb(rows, cols) = true;
%     % falsify boundaries
%     sampleMask_jb(1:LOFilterWH_half(2), :) = false;
%     sampleMask_jb(imgWH(2)-LOFilterWH_half(2):end, :) = false;
%     sampleMask_jb(:, 1:LOFilterWH_half(1)) = false;
%     sampleMask_jb(:, imgWH(1)-LOFilterWH_half(1):end, :) = false;
%         
%     % construct meta data
%     [rows, cols] = find(sampleMask_jb);
%     xy = [cols'; rows']; % be careful the order
%     ixy(:, startInd:startInd+size(xy, 2)-1) = [iInd*ones(1, size(xy, 2)); xy];
%     
%     % construct label
%     linInd = sub2ind(size(curLabel.cls), xy(2, :), xy(1, :));
%     curLabel = curLabel.cls(linInd);
%     label(startInd:startInd+size(xy, 2)-1) = curLabel;
%     
%     % update a pointer
%     startInd = startInd+size(xy, 2);
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
