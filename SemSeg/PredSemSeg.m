function [ o_cls, o_dist, o_params ] = PredSemSeg( i_imgs, i_mdls, i_params )
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
step = 1;

ixy = zeros(3, nData_approx);
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
    
    % extract features
    i_params.feat.sampleMask = sampleMask;
    [feat, tbParams] = GetDenseFeature(i_imgs(iInd), {'TextonBoostInt'}, i_params.feat);
    
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
dist = PredSemSeg_mex(x_meta, i_mdls, JBParams);

%% return
[~, o_cls] = max(dist, [], 2);
o_dist = dist;
o_params = struct('feat', tbParams, 'classifier', JBParams);
end

