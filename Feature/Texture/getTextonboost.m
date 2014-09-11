function [ o_feat ] = getTextonboost( i_tbParams, i_input, i_x_meta )
%GETTEXTONBOOST Summary of this function goes here
%   Detailed explanation goes here
%   i_tbParams.parts(:, i)      ith rectangle in the form of [xmin; xmax; ymin; ymax] 

parts = i_tbParams.parts;
nParts = size(parts, 2);
nTextons = size(i_tbParams.textons, 2);

% lfWH = i_tbParams.layoutFilterWH;
% lfWH = lfWH - mod(lfWH, 2);
% imgWH = [size(i_textonizedIntImgs, 2)-1; size(i_textonizedIntImgs, 1)-1];

%% check inputs
% assert(any(i_xy - lfWH/2 >= 1) && any(i_xy + lfWH/2 <= imgWH));

%% extract a feature
o_feat = zeros(nParts*nTextons, 1);
parfor (fInd=1:nParts*nTextons, 32)
% for fInd=1:nParts*nTextons
%     o_feat(fInd) = getithTextonboost(i_tbParams, i_input, i_imgInd, i_xys, fInd);
    o_feat(fInd) = getithTextonboost( i_tbParams, i_input, i_x_meta, fInd );
end
end

