function [ o_feat, o_params ] = GetTextonBoost( i_textIntImgs, i_params )
%GETTEXTONBOOST Summary of this function goes here
%   Detailed explanation goes here
%   i_tbParams.parts(:, i)      ith rectangle in the form of [xmin; xmax; ymin; ymax] 

nParts = i_params.nPart;
nTextons = i_params.nTexton;
nImgs = numel(i_textIntImgs);
sampleMask = i_params.sampleMask;

%% construct an index
[rows, cols] = find(sampleMask);
cols = unique(cols);
rows = unique(rows);
[xs, ys, is] = meshgrid(cols, rows, 1:nImgs); % be careful the order
ixy = [is(:)'; xs(:)'; ys(:)'];

%% extract a feature
if i_params.verbosity >= 1
    fprintf('* extract TextonBoost features for %d points...', size(ixy, 2));
    tbTic = tic;
end

feat_lin = zeros(nParts*nTextons, size(ixy, 2));
for fInd=1:nParts*nTextons
    feat_lin(fInd, :) = GetithTextonBoost( i_textIntImgs, ixy, fInd, i_params );
end
if i_params.verbosity >= 1
    fprintf('%s sec.\n', num2str(toc(tbTic)));
end
%% change the format
feat = zeros(numel(rows), numel(cols), nParts*nTextons, nImgs);
for iInd=1:nImgs
    feat_lin_cur = feat_lin(:, ixy(1, :) == iInd)';
    feat_lin_cur = reshape(feat_lin_cur, [size(feat, 1), size(feat, 2), size(feat, 3)]);
    feat(:, :, :, iInd) = feat_lin_cur;
end

%% return
o_feat = feat;
o_params = i_params;
o_params.sampleMask = sampleMask;
end
