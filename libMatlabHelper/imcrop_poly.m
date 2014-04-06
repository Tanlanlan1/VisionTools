function [ o_I, o_poly ] = imcrop_poly( i_I, i_poly, i_in )
%IMCROP_POLY Summary of this function goes here
%   i_I: [0, 1] image
%   i_poly(1, :): x coordinates of the polygon 
%   i_poly(2, :): y coordinates of the polygon 

i_I = im2double(i_I);
o_I = zeros(size(i_I));
o_poly = i_poly;

polyMask = poly2mask(i_poly(1, :), i_poly(2, :), size(i_I, 1), size(i_I, 2));
if i_in == false
    polyMask = ~polyMask;
end

if numel(size(i_I)) == 2
    dim = 1;
else
    dim = size(i_I, 3);
end

for d=1:dim
    tmp = i_I(:, :, d);
    tmp(~polyMask) = 0;
    o_I(:, :, d) = tmp;
end

if i_in == true
    xmin = min(i_poly(1, :));
    xmax = max(i_poly(1, :));
    ymin = min(i_poly(2, :));
    ymax = max(i_poly(2, :));
    
    o_I = imcrop(o_I, [xmin, ymin, xmax-xmin+1, ymax-ymin+1]);
    
    o_poly = bsxfun(@plus, i_poly, -[xmin; ymin]);
end

end

