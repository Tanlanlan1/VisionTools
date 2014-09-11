% 
%   GetSuperpixel demo
%
% ----------
% Written by Sangdon Park (sangdonp@cis.upenn.edu), 2014.
% All rights reserved.
%

%% load an image
img = im2double(imread('classroom.jpeg'));

%% obtain superpixel using NCut
label = GetSuperpixel(img, 'NCut', struct('N', 128));