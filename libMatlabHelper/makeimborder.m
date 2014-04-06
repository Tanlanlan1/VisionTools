function [ o_img ] = makeimborder( i_img, i_size, i_color )
%MAKEIMBORDER Summary of this function goes here
%   Detailed explanation goes here
o_img = i_img;
o_img(1:1+i_size-1, :, :) = i_color;
o_img(end-i_size+1:end, :, :) = i_color;
o_img(:, 1:1+i_size-1, :) = i_color;
o_img(:, end-i_size+1:end, :) = i_color;
end

