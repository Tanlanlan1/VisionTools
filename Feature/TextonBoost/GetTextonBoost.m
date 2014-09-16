function [ o_feat, o_params ] = GetTextonBoost( i_textIntImgs, i_params )
%GETTEXTONBOOST Summary of this function goes here
%   Detailed explanation goes here
%   i_tbParams.parts(:, i)      ith rectangle in the form of [xmin; xmax; ymin; ymax] 

nParts = i_params.nPart;
nTextons = i_params.nTexton;
LOFilterWH_half = (i_params.LOFilterWH-1)/2;
imgWH = [size(i_textIntImgs, 2); size(i_textIntImgs, 1)]-1;
nImg = size(i_textIntImgs, 4);
if ~isfield(i_params, 'sampleMask')
    i_params.sampleMask = true(imgWH(2), imgWH(1));
end
sampleMask = i_params.sampleMask;

%% construct an index
% falsify boundaries
sampleMask(1:LOFilterWH_half(2), :) = false;
sampleMask(imgWH(2)-LOFilterWH_half(2):end, :) = false;
sampleMask(:, 1:LOFilterWH_half(1)) = false;
sampleMask(:, imgWH(1)-LOFilterWH_half(1):end, :) = false;
% construct
[rows, cols] = find(sampleMask);
cols = unique(cols);
rows = unique(rows);
[xs, ys, is] = meshgrid(cols, rows, 1:nImg); % be careful the order
ixy = [is(:)'; xs(:)'; ys(:)'];

%% extract a feature
feat_lin = zeros(nParts*nTextons, size(ixy, 2));
for fInd=1:nParts*nTextons
%     feat_lin(fInd, :) = GetithTextonBoost_mex( i_textIntImgs, ixy, fInd, i_params.parts, i_params.LOFilterWH, i_params.nTexton );
    feat_lin(fInd, :) = GetithTextonBoost( i_textIntImgs, ixy, fInd, i_params );
end

%% change the format
feat = zeros(numel(rows), numel(cols), nParts*nTextons, nImg);
for iInd=1:nImg
    feat_lin_cur = feat_lin(:, ixy(1, :) == iInd)';
    feat_lin_cur = reshape(feat_lin_cur, [size(feat, 1), size(feat, 2), size(feat, 3)]);
    feat(:, :, :, iInd) = feat_lin_cur;
end

%% return
o_feat = feat;
o_params = i_params;
o_params.sampleMask = sampleMask;
end
