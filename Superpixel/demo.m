% 
%   GetSuperpixel demo
%
% ----------
% Written by Sangdon Park (sangdonp@cis.upenn.edu), 2014.
% All rights reserved.
%

%% load an image
% img = imresize(im2double(imread('classroom.jpeg')), 0.25);
img = imresize(im2double(imread('classroom.jpeg')), 0.25);

%% obtain superpixel using NCut
% label = GetSuperpixel(img, 'NCut', struct('N', 128));

label = GetSuperpixel(img, 'SLIC');