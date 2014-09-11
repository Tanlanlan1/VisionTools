
%% read an image
img = imread('tree-stump-texture.jpg');

%% extract features
feats = GetDenseFeature(img, {'color_Lab', 'texture_LM'}, struct('verbosity', 1));
