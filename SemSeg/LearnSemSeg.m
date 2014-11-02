function [ o_mdl, o_params, o_feats ] = LearnSemSeg( i_imgs, i_labels, i_params )
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
thisFilePath = fileparts(mfilename('fullpath'));
addpath(genpath([thisFilePath '/../Feature'])); %%FIXME
addpath(genpath([thisFilePath '/../JointBoost'])); %%FIXME

assert(isfield(i_imgs, 'img'));
assert(size(i_imgs, 1) == 1);

if ~isfield(i_params, 'pad')
    i_params.pad = false;
end
if ~isfield(i_params, 'nPerClsSample')
    i_params.nPerClsSample = 100;
end
if ~isfield(i_params, 'verbosity')
    i_params.verbosity = 0;
end
i_params.classifier.nPerClsSample = i_params.nPerClsSample;

nImgs = numel(i_imgs);
% nPerClsSample = i_params.nPerClsSample;
tbParams = i_params.feat;
% nCls = i_params.classifier.nCls;

%% learn a classifier (JointBoost)

% buid meta data for FeatCBFunc and labels
LOFilterWH_half = (i_params.feat.LOFilterWH-1)/2; 

% pad
if i_params.pad
    for iInd=1:nImgs
        % skip padding if there are precomputed results
        if ~(isfield(i_imgs(iInd), 'Texton') || isfield(i_imgs(iInd), 'TextonBoostInt'))
            i_imgs(iInd).img = padarray(i_imgs(iInd).img, [LOFilterWH_half(2) LOFilterWH_half(1) 0], 'symmetric', 'both');
        end
        i_labels(iInd).cls = padarray(i_labels(iInd).cls, [LOFilterWH_half(2) LOFilterWH_half(1) 0], 'symmetric', 'both');
        
    end
end
assert(size(i_labels(1).cls, 1) == size(i_imgs(1).img, 1));
assert(size(i_labels(1).cls, 2) == size(i_imgs(1).img, 2));

% extract Texton
[feats_texton, tbParams] = GetDenseFeature(i_imgs, {'Texton'}, tbParams); %%FIXME: sampling points are different with JointBoost

% if i_params.binary == 1 
%     nData = nPerClsSample*2;
%     ixy = zeros(3, nData, nCls-1);
%     label = zeros(nData, 1, nCls-1);
% else
%     nData = nPerClsSample*nCls;
%     ixy = zeros(3, nData, 1);
%     label = zeros(nData, 1, 1);
% end

ixy = cell(1, nImgs);
label = cell(nImgs, 1);

% startInd = 1;
feats_textInt = [];
for iInd=1:nImgs
    
    curImg = feats_texton(iInd);
%     curLabel = i_labels(iInd);
    imgWH = [size(curImg.img, 2); size(curImg.img, 1)];
    
    % extract feature (TextonBoost) 
    [curOut, tbParams] = GetDenseFeature(curImg, {'TextonBoostInt'}, tbParams); 
    feats_textInt = [feats_textInt; curOut]; %%FIXME: ugly. do that outside of the loop
    
%     % falsify boundaries
%     curLabel.cls(1:LOFilterWH_half(2), :) = nan;
%     curLabel.cls(imgWH(2)-LOFilterWH_half(2):end, :) = nan;
%     curLabel.cls(:, 1:LOFilterWH_half(1)) = nan;
%     curLabel.cls(:, imgWH(1)-LOFilterWH_half(1):end, :) = nan;
    
    % sampling data
    [ixy{iInd}, label{iInd}] = sampleData(i_params, iInd, [size(curImg.img, 2); size(curImg.img, 1)], i_labels(iInd));
    
%     % build sampleMask %%FIXME: should be balanced across images also...
%     sampleMask_jb = zeros(imgWH(2), imgWH(1)); 
%     for cInd=0:nCls % include bg=0
%         [rows, cols] = find(curLabel.cls == cInd);
%         if isempty(rows)
%             continue;
%         end
%         
%         rndInd = randi(numel(rows), [nPerClsSample, 1]);
%         linInd = sub2ind(size(sampleMask_jb), rows(rndInd), cols(rndInd));
%         for tmp=1:numel(linInd) % should be updated sequencially
%             sampleMask_jb(linInd(tmp)) = sampleMask_jb(linInd(tmp)) + 1;
%         end
%     end
%     if i_params.verbosity >= 2
%         figure(2000); 
%         imagesc(sampleMask_jb);
%     end
%     
%     % construct pixel location matrix
%     xys = [];
%     while true     
%         [rows, cols] = find(sampleMask_jb>=1);
%         if isempty(rows)
%             break;
%         end
%         sampleMask_jb(sampleMask_jb>=1) = sampleMask_jb(sampleMask_jb>=1) - 1;
%         
%         xy = [cols'; rows']; % be careful the order
%         xys = [xys xy];    
%     end
%     
%     % construct label
%     linInd = sub2ind(size(curLabel.cls), xys(2, :), xys(1, :));
%     label = curLabel.cls(linInd);
%     
%     % update ixy and label
%     if i_params.binary == 1 % select bg for each classes
%         for cInd=1:nCls-1 % exclude bg
%             
%         end
%     else
%         ixy(:, startInd:startInd+size(xys, 2)-1) = [iInd*ones(1, size(xys, 2)); xys];
%         label(startInd:startInd+size(xys, 2)-1) = label;
%     end
%     
%     % update a pointer
%     startInd = startInd+size(xys, 2);
end
% ixy(:, startInd:end, :) = [];
% label(startInd:end, 1, :) = [];

ixy = cell2mat(ixy);
label = cell2mat(label);
x_meta = struct('ixy', ixy, 'intImgFeat', feats_textInt, 'TBParams', tbParams);

% run
JBParams = i_params.classifier;
JBParams.nData = numel(label);
if JBParams.verbosity >= 1
    fprintf('* Train %d data\n', JBParams.nData);
end
% check input parameters
[x_meta_mex, label_mex, JBParams_mex] = convParamsType(x_meta, label, JBParams);
if JBParams.verbosity >= 1
    mexTID = tic;
end
mdls = LearnSemSeg_mex(x_meta_mex, label_mex, JBParams_mex);
if JBParams.verbosity >= 1
    fprintf('* Running time LearnSemSeg_mex: %s sec.\n', num2str(toc(mexTID)));
end

%% return
o_mdl = mdls; 
o_params = i_params;
o_params.feat = tbParams;
o_params.classifier = JBParams;
o_feats = feats_textInt;
end

function [o_ixy, o_label] = sampleData(i_params, iInd, imgWH, i_label)

nPerClsSample = i_params.nPerClsSample;
nCls = i_params.classifier.nCls;
LOFilterWH_half = (i_params.feat.LOFilterWH-1)/2;

% falsify boundaries
i_label.cls(1:LOFilterWH_half(2), :) = nan;
i_label.cls(imgWH(2)-LOFilterWH_half(2):end, :) = nan;
i_label.cls(:, 1:LOFilterWH_half(1)) = nan;
i_label.cls(:, imgWH(1)-LOFilterWH_half(1):end, :) = nan;

% build sampleMask %%FIXME: should be balanced across images also...
sampleMask_jb = zeros(imgWH(2), imgWH(1)); 
for cInd=0:nCls % include bg=0
    [rows, cols] = find(i_label.cls == cInd);
    if isempty(rows)
        continue;
    end

    rndInd = randi(numel(rows), [nPerClsSample, 1]);
    linInd = sub2ind(size(sampleMask_jb), rows(rndInd), cols(rndInd));
    for tmp=1:numel(linInd) % should be updated sequencially
        sampleMask_jb(linInd(tmp)) = sampleMask_jb(linInd(tmp)) + 1;
    end
end
if i_params.verbosity >= 2
    figure(2000); 
    imagesc(sampleMask_jb);
end

% construct pixel location matrix
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
ixys = [iInd*ones(1, size(xys, 2)); xys];

% construct label
linInd = sub2ind(size(i_label.cls), xys(2, :), xys(1, :));
label = i_label.cls(linInd);

%% return ixy and label
if i_params.classifier.binary == 1 % select bg for each classes
    o_ixy = zeros(3, nPerClsSample*2, nCls);
    o_label = zeros(nPerClsSample*2, 1, nCls);
    for cInd=1:nCls
        posInd = label == cInd;
        assert(sum(posInd) == nPerClsSample);
        % add positives
        o_ixy(:, 1:nPerClsSample, cInd) = ixys(:, posInd);
        o_label(1:nPerClsSample, 1, cInd) = 1; % pos
        % add negatives
        [rs, cs] = find(i_label.cls ~= cInd & ~isnan(i_label.cls));
        rndInd = randi(numel(rs), [1, nPerClsSample]);
        o_ixy(:, nPerClsSample+1:end, cInd) = [iInd*ones(1, nPerClsSample); cs(rndInd)'; rs(rndInd)'];
        o_label(nPerClsSample+1:end, 1, cInd) = 2; % neg
    end
else
    o_ixy = ixys;
    o_label = label;
end
 
end