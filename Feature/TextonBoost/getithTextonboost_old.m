function [ o_ithfeat ] = getithTextonboost( i_tbParams, i_input, i_x_meta, i_i )
%GETTEXTONBOOST Summary of this function goes here
%   Detailed explanation goes here
%   i_tbParams.parts(:, i)      ith rectangle in the form of [xmin; xmax; ymin; ymax] 


parts = i_tbParams.parts;
% nParts = size(parts, 2);
lfWH = i_tbParams.layoutFilterWH;
lfWH = lfWH - mod(lfWH, 2);
nData = size(i_x_meta, 2);
nTextons = size(i_tbParams.textons, 2);

imgWH = [size(i_input.data(1).textIntImgs, 2)-1; size(i_input.data(1).textIntImgs, 1)-1];

o_ithfeat = zeros(size(i_x_meta, 2), 1, 'single');
pInd = floor((i_i-1)/nTextons)+1;
tInd = i_i - (pInd-1)*nTextons;    
    
parfor (dInd=1:nData, 32)
% for dInd=1:nData
    
%     xys = i_x_meta(2:4, dInd);
%     curScale = i_input.scales(i_x_meta(4, dInd));


    %% extract a feature
    
%     normalizedxy = round(xys(1:2)/curScale);
%     normalizedWH = round(lfWH/curScale);


    normalizedPart = round(parts(:, pInd)/i_input.scales(i_x_meta(4, dInd)));
    
%     lf_tl = normalizedxy-round(normalizedWH/2);
    lf_tl = round(i_x_meta(2:3, dInd)/i_input.scales(i_x_meta(4, dInd)) - lfWH/i_input.scales(i_x_meta(4, dInd))/2);

    xy_part_tl = min(max([normalizedPart(1); normalizedPart(3)] + lf_tl - 1, [1; 1]), imgWH); % check bounary conditions
    xy_part_br = min(max([normalizedPart(2); normalizedPart(4)] + lf_tl - 1, [1; 1]), imgWH); % check bounary conditions
    partArea = (xy_part_br(1) - xy_part_tl(1) + 1)*(xy_part_br(2) - xy_part_tl(2) + 1);
    

    o_ithfeat(dInd) = ...
        (single(i_input.data(i_x_meta(1, dInd)).textIntImgs(xy_part_br(2) + 1, xy_part_br(1) + 1, tInd))...
        - single(i_input.data(i_x_meta(1, dInd)).textIntImgs(xy_part_tl(2), xy_part_br(1) + 1, tInd))...
        - single(i_input.data(i_x_meta(1, dInd)).textIntImgs(xy_part_br(2) + 1, xy_part_tl(1), tInd))...
        + single(i_input.data(i_x_meta(1, dInd)).textIntImgs(xy_part_tl(2), xy_part_tl(1), tInd)))...
        /(partArea+eps);

end

end

