function [ o_im ] = myim2double( i_im )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

o_im = im2double(i_im);
o_im(o_im > 1) = 1;
o_im(o_im < 0) = 0;
end

