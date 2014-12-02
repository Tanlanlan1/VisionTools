function [ o_label_xymean ] = FindSegMeanPos( i_segLabels )
%FINDSEGMEANPOS Summary of this function goes here
%   Detailed explanation goes here

ID2Lbl = unique(i_segLabels(:)');
o_label_xymean = zeros(2, numel(ID2Lbl));
parfor lInd=1:numel(ID2Lbl) %%FIXME: inefficient, need mex implementation
    mask = i_segLabels == ID2Lbl(lInd);
    [rs, cs] = find(mask);
    o_label_xymean(:, lInd) = round([mean(cs); mean(rs)]);
end
        
end

