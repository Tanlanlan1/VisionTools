function [ o_mat ] = crop3dmat( i_mat, i_rect)
%CROP3DMAT Summary of this function goes here
%   Detailed explanation goes here

if (size(i_mat, 2) == i_rect(3)-i_rect(1)+1) && (size(i_mat, 1) == i_rect(4)-i_rect(2)+1)
    o_mat = i_mat;
    return;
end

assert(2 <= numel(size(i_mat)) && numel(size(i_mat)) <= 3);

% nW = size(i_mat, 2);
% nH = size(i_mat, 1);
% if numel(size(i_mat)) == 2
%     nD = 1;
% else
%     nD = size(i_mat, 3);
% end

sPnt = i_rect(1:2);
ePnt = i_rect(1:2) + i_rect(3:4) - 1;

o_mat = i_mat(sPnt(2):ePnt(2), sPnt(1):ePnt(1), :);

% [xs, ys] = meshgrid(sPnt(1):ePnt(1), sPnt(2):ePnt(2));
% linInd = sub2ind([nH, nW], ys, xs);
% linIndAll = bsxfun(@plus, repmat(linInd(:), [1, nD]), [0 cumsum(ones(1, nD-1)*nW*nH)]);
% o_mat = reshape(i_mat(linIndAll(:)), [ePnt(2)-sPnt(2)+1 ePnt(1)-sPnt(1)+1 nD]);

end

