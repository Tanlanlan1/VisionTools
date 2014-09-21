
%% read an image
img = imread('tree-stump-texture.jpg');

% %% extract features
% feats = GetDenseFeature(img, {'color_Lab', 'texture_LM'}, struct('verbosity', 1));

% %% extract texton features
% feats = GetDenseFeature(img, {'texton'}, struct('nTexton', 64, 'verbosity', 2));

%% extract TextonBoost
sampleMask = false(size(img, 1), size(img, 2));
cr = 1:10:size(sampleMask, 2);
rr = 1:10:size(sampleMask, 1);
[rows, cols]  = meshgrid(rr, cr);
linind = sub2ind(size(sampleMask), rows, cols);
sampleMask(linind) = true;

imgs(1).img = img;
feats = GetDenseFeature(imgs, {'TextonBoost'}, ...
    struct('nTexton', 64, 'nPart', 16, 'LOFilterWH', [13; 13], 'sampleMask', sampleMask,'verbosity', 2));