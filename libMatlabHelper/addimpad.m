function [ o_img ] = addimpad( i_img, i_padSz )
%ADDIMPAD Summary of this function goes here
%   Detailed explanation goes here
if i_padSz == 0
    o_img = i_img;
    return;
end
o_img = padarray(i_img, [i_padSz i_padSz]);
end

