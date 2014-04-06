function [i_img] = getPascalImg(i_params, i_anno)

if ~isfield(i_anno, 'img') || isempty(i_anno.img)
    curImFN = sprintf('%s/%s.jpg', i_params.db.imgDir, i_anno.filename(1:end-4));
    i_img = imread(curImFN);
else
    i_img = i_anno.img;
end

i_img = im2double(gray2rgb(i_img));
end