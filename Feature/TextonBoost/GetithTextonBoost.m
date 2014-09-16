function [ o_ithfeat ] = GetithTextonBoost( i_textIntImgs, i_ixy, i_i, i_params )
%GETTEXTONBOOST Summary of this function goes here
%   Detailed explanation goes here
%   i_tbParams.parts(:, i)      ith rectangle in the form of [xmin; xmax; ymin; ymax] 

o_ithfeat = GetithTextonBoost_mex( i_textIntImgs, i_ixy, i_i, i_params.parts, i_params.LOFilterWH, i_params.nTexton );

% %% init
% parts = i_params.parts;
% LOFWH = i_params.LOFilterWH;
% nTexton = i_params.nTexton;
% nData = size(i_ixy, 2);
% 
% %% extract
% pInd = floor((i_i-1)/nTexton)+1;
% tInd = i_i - (pInd-1)*nTexton;
% 
% o_ithfeat = zeros(nData, 1);
% for dInd=1:nData
%     ixy = i_ixy(:, dInd);
%     part = parts(:, pInd);
%     LOF_tl = ixy(2:3) - (LOFWH-1)/2;
% 
%     xy_part_tl = LOF_tl + [part(1); part(3)] - 1;
%     xy_part_br = LOF_tl + [part(2); part(4)] - 1;
% 
%     partArea = (xy_part_br(1) - xy_part_tl(1) + 1)*(xy_part_br(2) - xy_part_tl(2) + 1);   
%     curIntImg = i_textIntImgs(:, :, tInd, ixy(1));
%     o_ithfeat(dInd) = (...
%         curIntImg(   xy_part_br(2) + 1,  xy_part_br(1) + 1  )...
%         - curIntImg( xy_part_tl(2),      xy_part_br(1) + 1  )...
%         - curIntImg( xy_part_br(2) + 1,  xy_part_tl(1)      )...
%         + curIntImg( xy_part_tl(2),      xy_part_tl(1)      ))/(partArea+eps);
% end
end

