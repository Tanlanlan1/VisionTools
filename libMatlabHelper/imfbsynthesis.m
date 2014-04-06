function [ o_syn, o_poly ] = imfbsynthesis( i_fg, i_bg, i_fgPoly, i_fgLoc )
%IMFBSYNTHESIS synthesize the fg image in the bg image at the given location
%   i_fgPoly(:, i): (x, y)' of the i-th point

imW = size(i_bg, 2);
imH = size(i_bg, 1);

%% choose random location
xmin_fgo = min(i_fgPoly(1, :));
xmax_fgo = max(i_fgPoly(1, :));
ymin_fgo = min(i_fgPoly(2, :));
ymax_fgo = max(i_fgPoly(2, :));

fgObjW = round(xmax_fgo - xmin_fgo);
fgObjH = round(ymax_fgo - ymin_fgo);

assert(imW-fgObjW+1>=1);
assert(imH-fgObjH+1>=1);

if isempty(i_fgLoc)
    fgX = randi([1 imW-fgObjW+1], 1);
    fgY = randi([1 imH-fgObjH+1], 1);
else
    fgX = i_fgLoc(1);
    fgY = i_fgLoc(2);
end

%% 
[fg_crop, poly_crop] = imcrop_poly(i_fg, i_fgPoly, true);
fgImg_rnd = fg_crop;
fgImg_rnd = padarray(fgImg_rnd, [fgY-1 fgX-1], 0, 'pre');


fgImg_rnd = padarray(fgImg_rnd, [imH-fgObjH-fgY+1 imW-fgObjW-fgX+1], 0, 'post');
fgPoly_rnd = bsxfun(@plus, poly_crop, [fgX-1; fgY-1]);

bg_crop = imcrop_poly(i_bg, fgPoly_rnd, false);
bg_crop = imresize(bg_crop, [size(fgImg_rnd, 1), size(fgImg_rnd, 2)]); %% not good!

o_syn = imadd(fgImg_rnd, bg_crop);
o_poly = fgPoly_rnd;

end

