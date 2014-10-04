function [ o_prop ] = GetObjectProposal( i_imgs, i_params )
% 
%   Matlab wrapper of obtaining object proposal based on the selective
%   search [1]
%   
% ----------
%   Input: 
% 
%       i_imgs:         a image or a cell array of images
%       i_params:
%           k           a scale of a base segmentation (default: 100)
%           sigma       a smoothing filter size before segmentation (default: 0.8)
%           minRatio    a minimum pixel ratio of the final proposals (default: 0.2)           
%           maxRatio    a maximum pixel ratio of the final proposals (default: 0.8)    
%           nmsOverlap  a overlap ratio used in NMS. bb is suppressed if
%                       over this threshold (default: 0.5)
%           verbosity   a verbose level (default: 0, silent)
% 
% ----------
%   Output:
% 
%       o_prop:         object proposals in a Nx1 cell array, where N is
%                       the number of an image
%       
% ----------
%   Dependency:
%   
% ----------
%   Reference:
% 
%       [1] Jasper R. R. Uijlings, Koen E. A. van de Sande, Theo Gevers,
%       Arnold W. M. Smeulders, Selective Search for Object Recognition.
%       IJCV 2013.
%
% ----------
% Written by Sangdon Park (sangdonp@cis.upenn.edu), 2014.
% All rights reserved.
%

%% init
% use Selective Search
thisFilePath = fileparts(mfilename('fullpath'));
addpath(genpath([thisFilePath '/../SelectiveSearch']));
SSinit;

if nargin < 2
    i_params = struct('verbosity', 0);
end
if ~isfield(i_params, 'verbosity')
    i_params.verbosity = 0;
end
if ~isfield(i_params, 'k')
    i_params.k = 100;
end
if ~isfield(i_params, 'sigma')
    i_params.sigma = 0.8;
end
if ~isfield(i_params, 'minRatio')
    i_params.minRatio = 0.001;
end
if ~isfield(i_params, 'maxRatio')
    i_params.maxRatio = 0.1;
end
if ~isfield(i_params, 'nmsOverlap')
    i_params.nmsOverlap = 0.5;
end
if ~isfield(i_params, 'bounaryRatio')
    i_params.bounaryRatio = 0.2;
end

if iscell(i_imgs)
    nImg = numel(i_imgs);
    cellFlag = true;
else
    i_imgs = {i_imgs};
    nImg = 1;
    cellFlag = false;
end
%% set params

% Parameters. Note that this controls the number of hierarchical
% segmentations which are combined.
colorTypes = {'Hsv', 'Lab', 'RGI', 'H', 'Intensity'}';
colorType = colorTypes{2};
% Here you specify which similarity functions to use in merging
simFunctionHandles = {@SSSimColourTextureSizeFillOrig, @SSSimTextureSizeFill, @SSSimBoxFillOrig, @SSSimSize};
% simFunctionHandles = {@SSSimColourTextureSizeFillOrig};

%% Perform Selective Search
o_prop = cell(nImg, 1);
parfor iInd=1:nImg
    im = im2double(i_imgs{iInd});
    minSize = i_params.k;
    nPix = size(im, 1)*size(im, 2);
    
    % run
    [boxes, blobIndIm, blobBoxes, hierarchy] = Image2HierarchicalGrouping(im, i_params.sigma, i_params.k, minSize, colorType, simFunctionHandles);
    boxes = BoxRemoveDuplicates(boxes);
    % show
    if i_params.verbosity > 2
        ShowRectsWithinImage(boxes, 5, 5, im);
    end
    
    % obtain all possible blobs
    hBlobs = cell(numel(hierarchy), 1);
    for hInd=1:numel(hierarchy)
        hBlobs{hInd} = RecreateBlobHierarchyIndIm(blobIndIm, blobBoxes, hierarchy{hInd});
        % show
        if i_params.verbosity > 1
            ShowBlobs(hBlobs{hInd}, 5, 5, im);
        end
    end
    
    % post-process to obtain proper sized proposals
    hBlobs_pruned = cell(numel(hierarchy), 1);
    for hInd=1:numel(hierarchy)
        curBlob = hBlobs{hInd};
        valid = false(numel(curBlob), 1);
        for bInd=1:numel(curBlob)
            nCurPix = sum(curBlob{bInd}.mask(:));
            if i_params.minRatio*nPix <= nCurPix && i_params.maxRatio*nPix >= nCurPix
                valid(bInd) = true;
            end
        end
        hBlobs_pruned{hInd} = curBlob(valid);
        % show
        if i_params.verbosity > 1
            ShowBlobs(hBlobs_pruned{hInd}, 5, 5, im);
        end
    end
    blobs = hBlobs_pruned{:};
    blobs = cell2mat(blobs);
    
    % change the rect format % [x y w h] and add convexity
    for bInd=1:numel(blobs)
        % [x y w h]
        blobs(bInd).rect_matlab = [blobs(bInd).rect(2) blobs(bInd).rect(1) blobs(bInd).rect(4)-blobs(bInd).rect(2) blobs(bInd).rect(3)-blobs(bInd).rect(1)];
        blobs(bInd).convexity = sum(blobs(bInd).mask(:))/(blobs(bInd).rect_matlab(3)*blobs(bInd).rect_matlab(4));
    end
    
    % post-process to remove blobs at the boundary
    xbndlen = round(i_params.bounaryRatio*size(im, 2));
    ybndlen = round(i_params.bounaryRatio*size(im, 1));
    xmin = round(xbndlen/2);
    xmax = size(im, 2)-round(xbndlen/2);
    ymin = round(ybndlen/2);
    ymax = size(im, 1)-round(ybndlen/2);
    valid = true(numel(blobs), 1);
    for bInd=1:numel(blobs)
        cx = blobs(bInd).rect_matlab(1) + round(blobs(bInd).rect_matlab(3)/2);
        cy = blobs(bInd).rect_matlab(2) + round(blobs(bInd).rect_matlab(4)/2);
        if cx < xmin || cx > xmax || cy < ymin || cy > ymax
            valid(bInd) = false;
        end
    end
    blobs = blobs(valid);
    
    % suppress duplicated blobs
    prop = suppressBlobs(blobs, i_params.nmsOverlap);
    
    % show
    if i_params.verbosity >= 1
        prop_cell = mat2cell(prop, ones(1, numel(prop)), 1);
        ShowBlobs(prop_cell, 5, 5, im);
    end
    
    %% return
    o_prop{iInd} = prop;
    
end

if ~cellFlag
    o_prop = cell2mat(o_prop);
else
    o_prop = reshape(o_prop, size(i_imgs));
end
end

function [o_blobs] = suppressBlobs(i_blobs, i_overlap)
% suppress blobs based on the number of pixels of blobs. a convex blob is preferred. Check a overlap simply based on a bounding box.

[~, I] = sort([i_blobs(:).convexity], 'descend');
blobs = i_blobs(I);
valid = true(numel(blobs), 1);
for bInd1=1:numel(blobs)
    bb1 = blobs(bInd1).rect_matlab;

    for bInd2=bInd1+1:numel(blobs)
        if ~valid(bInd2)
            continue;
        end
        bb2 = blobs(bInd2).rect_matlab;
        
        % simply check overlap between bounding boxes
%         intArea = rectint(bb1, bb2);
        intArea = myrectint(bb1, bb2);
        uniArea = bb1(3)*bb1(4) + bb2(3)*bb2(4) - intArea;
        ov = intArea/uniArea;
        % suppress
        if ov > i_overlap
            valid(bInd2) = false;
        end
    end
end

%% return
o_blobs = blobs(valid);
[~, I] = sort([o_blobs(:).convexity], 'descend');
o_blobs = o_blobs(I);
end

function [out] = myrectint(A, B)
% modified matlab code for efficiency
assert(size(A, 1) == 1 && size(B, 1) == 1);

leftA = A(:,1);
bottomA = A(:,2);
rightA = leftA + A(:,3);
topA = bottomA + A(:,4);

leftB = B(:,1)';
bottomB = B(:,2)';
rightB = leftB + B(:,3)';
topB = bottomB + B(:,4)';

out = (max(0, min(rightA, rightB) - max(leftA, leftB))) .* ...
    (max(0, min(topA, topB) - max(bottomA, bottomB)));
end


