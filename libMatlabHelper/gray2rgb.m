function [ o_img ] = gray2rgb( i_img )
%GRAY2RGB Summary of this function goes here
%   Detailed explanation goes here

if numel(size(i_img)) == 2
   o_img = repmat(i_img, [1 1 3]);
else
    o_img = i_img;
end
end

