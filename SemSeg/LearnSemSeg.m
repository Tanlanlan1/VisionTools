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
assert(size(i_labels(1).cls, 1) == size(i_imgs(1).img, 1));
assert(size(i_labels(1).cls, 2) == size(i_imgs(1).img, 2));
assert(numel(i_imgs) == numel(i_labels));
assert(isfield(i_imgs, 'img'));

thisFilePath = fileparts(mfilename('fullpath'));
addpath(genpath([thisFilePath '/../Feature'])); %%FIXME
addpath(genpath([thisFilePath '/../JointBoost'])); %%FIXME

if ~isfield(i_params, 'pad') %%FIXME: not use?
    i_params.pad = false;
end
if ~isfield(i_params, 'nPerClsSample')
    i_params.nPerClsSample = 100;
end
if ~isfield(i_params, 'verbosity')
    i_params.verbosity = 0;
end
i_params.classifier.nCls = size(i_labels(1).cls, 3);
i_params.classifier.nPerClsSample = i_params.nPerClsSample;
tbParams = i_params.feat;

%% sample images for efficient computation
[i_imgs, i_labels] = sampleImages( i_imgs, i_labels);

%% extract features
% extract Texture
[feat_texture, tbParams] = GetDenseFeature(i_imgs, {'Texture_LM'}, tbParams);
% % sample images for extracting Texton
% [imgs_Texton] = sampleImgForTexton(feat_texture, i_labels);
% extract Texton
[~, tbParams] = GetDenseFeature(feat_texture, {'TextonInit'}, tbParams); %%FIXME: Texton sampling points are different with JointBoost
% extract Texton features
[feats_texton, tbParams] = GetDenseFeature(feat_texture, {'Texton'}, tbParams);
% init TextonBoost
[~, tbParams] = GetDenseFeature([], {'TextonBoostInit'}, tbParams); 
% extract TextonBoost: require pre-extracted Texton and part initialization
% feats_textInt = arrayfun(@(x) GetDenseFeature(x, {'TextonBoostInt'}, tbParams), feats_texton);
feats_textInt = GetDenseFeature(feats_texton, {'TextonBoostInt'}, tbParams);

% %% extract features
% % extract Texture
% [feat_texture, tbParams] = GetDenseFeature(i_imgs, {'Texture_LM'}, tbParams);
% % sample images for extracting Texton
% [imgs_Texton] = sampleImgForTexton(feat_texture, i_labels);
% % extract Texton
% [~, tbParams] = GetDenseFeature(imgs_Texton, {'TextonInit'}, tbParams); %%FIXME: Texton sampling points are different with JointBoost
% % extract Texton features
% [feats_texton, tbParams] = GetDenseFeature(feat_texture, {'Texton'}, tbParams);
% % init TextonBoost
% [~, tbParams] = GetDenseFeature([], {'TextonBoostInit'}, tbParams); 
% % extract TextonBoost: require pre-extracted Texton and part initialization
% % feats_textInt = arrayfun(@(x) GetDenseFeature(x, {'TextonBoostInt'}, tbParams), feats_texton);
% feats_textInt = GetDenseFeature(feats_texton, {'TextonBoostInt'}, tbParams);

%% sampling data
[ixy, label] = sampleData(i_params, i_labels);

%% learn a classifier (JointBoost)
% set input parameters
x_meta = struct('ixy', ixy, 'intImgFeat', feats_textInt, 'TBParams', tbParams);
JBParams = i_params.classifier;
JBParams.nData = sum(arrayfun(@(x) numel(x.labels_cls), label));
if JBParams.verbosity >= 1
    fprintf('* Train %d data\n', JBParams.nData);
end
[x_meta_mex, label_mex, JBParams_mex] = convParamsType(x_meta, label, JBParams);
if JBParams.verbosity >= 1
    mexTID = tic;
end
% learn
mdls = LearnSemSeg_mex(x_meta_mex, label_mex, JBParams_mex);
if JBParams.verbosity >= 1
    fprintf('* Running time LearnSemSeg_mex: %s sec.\n', num2str(toc(mexTID)));
end

%% return
o_mdl = mdls; 
o_params = i_params;
o_params.feat = tbParams;
o_params.classifier = JBParams;
o_feats = []; %%FIXME: remove later
end

function [o_imgs] = sampleImgForTexton(i_imgs, i_labels)
nImgs = numel(i_imgs);
nCls = size(i_labels(1).cls, 3);
% init
o_imgs = i_imgs(1); 
% find image ind with valid label
ind = 1;
for iInd=1:nImgs
    if any(any(sum(i_labels(iInd).cls(:, :, 1:nCls-1), 3)))
        o_imgs(ind) = i_imgs(iInd);
        ind = ind + 1;
    end
end
end

function [o_imgs, o_labels] = sampleImages(i_imgs, i_labels)
nImgs = numel(i_imgs);
nCls = size(i_labels(1).cls, 3);
% init
o_imgs = i_imgs(1); 
o_labels = i_labels(1);
% find image ind with valid label
ind = 1;
for iInd=1:nImgs
    if any(any(sum(i_labels(iInd).cls(:, :, 1:nCls-1), 3)))
        o_imgs(ind) = i_imgs(iInd);
        o_labels(ind) = i_labels(iInd);
        ind = ind + 1;
    end
end
end


function [o_ixys, o_labels] = sampleData(i_params, i_label)

nPerClsSample = i_params.nPerClsSample;
nCls = i_params.classifier.nCls;
nImgs = numel(i_label);

%% randomly select positive points for each classes
ixys = cell(1, nImgs);
labels = cell(nImgs, 1);
for iInd=1:nImgs
    ixys_i = cell(1, nCls);
    labels_i = cell(nCls, 1);
    for cInd=1:nCls % NOT include bg=0
        [rows, cols] = find(i_label(iInd).cls(:, :, cInd));
        if isempty(rows)
            continue;
        end

        rndInd = randi(numel(rows), [nPerClsSample, 1]);
        ixys_i{cInd} = [...
            iInd*ones(1, nPerClsSample); ...
            reshape(cols(rndInd), [1 nPerClsSample]); 
            reshape(rows(rndInd), [1 nPerClsSample])];
        labels_i{cInd} = ones(nPerClsSample, 1)*cInd;
    end
    ixys{iInd} = cell2mat(ixys_i);
    labels{iInd} = cell2mat(labels_i);
end
ixys = cell2mat(ixys);
labels = cell2mat(labels);
assert(size(ixys, 2) == numel(labels));
    
%% return ixys and labels
if i_params.classifier.binary == 1 % select bg for each classes
    % multiple binary classifiers
    
    % generate bg
    ixys_bg = cell(1, nImgs);
    labels_bg = cell(nImgs, 1);
    for iInd=1:nImgs
        ixys_i_bg = cell(1, nCls);
        labels_i_bg = cell(nCls, 1);
        for cInd=1:nCls % NOT include bg=0
            % check there are positive labels
            if ~any(any(i_label(iInd).cls(:, :, cInd)))
                continue;
            end
            % find bg pixels
            [rows, cols] = find(~i_label(iInd).cls(:, :, cInd));
            if isempty(rows)
                continue;
            end

            rndInd = randi(numel(rows), [nPerClsSample, 1]);
            ixys_i_bg{cInd} = [...
                iInd*ones(1, nPerClsSample); ...
                reshape(cols(rndInd), [1 nPerClsSample]); 
                reshape(rows(rndInd), [1 nPerClsSample])];
            labels_i_bg{cInd} = ones(nPerClsSample, 1)*cInd;
        end
        ixys_bg{iInd} = cell2mat(ixys_i_bg);
        labels_bg{iInd} = cell2mat(labels_i_bg);
    end
    ixys_bg = cell2mat(ixys_bg);
    labels_bg = cell2mat(labels_bg);
    
    % return
    o_ixys = struct('ixys_cls', []);
    o_ixys(nCls) = o_ixys;
    o_labels = struct('labels_cls', []);
    o_labels(nCls) = o_labels;
    for cInd=1:nCls
        posInd = labels == cInd;
        negInd = labels_bg == cInd;
        nPos = sum(posInd);
        nNeg = sum(negInd);
        if nPos == 0
            continue;
        end
        assert(nPos == nNeg);
        % init
        ixys_cls = zeros(3, nPos*2);
        labels_cls = zeros(nPos*2, 1);
        % add positives
        ixys_cls(:, 1:nPos) = ixys(:, posInd);
        labels_cls(1:nPos) = 1; % pos
        % add negatives
        ixys_cls(:, nPos+1:end) = ixys_bg(:, negInd);
        labels_cls(nPos+1:end) = 2; % neg
        
%         negCandiInd = find(~posInd);
%         rndInd = randi(numel(negCandiInd), [1, nPos]);
%         
% %         [rs, cs] = find(~i_label.cls(:, :, cInd));
% %         rndInd = randi(numel(rs), [1, nPos]);
% %         o_ixys{iInd}(:, nPerClsSample+1:end, cInd) = [iInd*ones(1, nPerClsSample); cs(rndInd)'; rs(rndInd)'];
% 
% 
%         ixys_cls(:, nPos+1:end) = ixys(:, negCandiInd(rndInd));
%         labels_cls(nPos+1:end) = 2; % neg

        % save
        o_ixys(cInd).ixys_cls = ixys_cls;
        o_labels(cInd).labels_cls = labels_cls;
        
%         % add positives
%         o_ixys{iInd}(:, 1:nPerClsSample, cInd) = ixys(:, posInd);
%         labels{iInd}(1:nPerClsSample, 1, cInd) = 1; % pos
%         % add negatives
%         [rs, cs] = find(~i_label.cls(:, :, cInd)); %& ~isnan(i_label.cls));
%         rndInd = randi(numel(rs), [1, nPerClsSample]);
%         o_ixys{iInd}(:, nPerClsSample+1:end, cInd) = [iInd*ones(1, nPerClsSample); cs(rndInd)'; rs(rndInd)'];
%         labels{iInd}(nPerClsSample+1:end, 1, cInd) = 2; % neg
    end
else
    % multiclass classifiers
    o_ixys = struct('ixys_cls', ixys);
    o_labels = struct('labels_cls', labels);
end
 
end