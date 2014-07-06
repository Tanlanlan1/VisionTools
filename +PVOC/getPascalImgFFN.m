function [ o_imgFFN ] = getPascalImgFFN( i_params, i_anno )
%GETPASCALIMGPATH Summary of this function goes here
%   Detailed explanation goes here
o_imgFFN = sprintf('%s/%s.jpg', i_params.db.imgDir, i_anno.filename(1:end-4));
end

