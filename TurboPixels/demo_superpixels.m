% Runs the superpixel code on the lizard image

addpath('lsmlib');
img = im2double(imread('lizard.jpg'));
[phi,boundary,disp_img] = superpixels(img, 500);
imagesc(disp_img);
