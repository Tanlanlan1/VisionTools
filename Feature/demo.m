
%% read an image
img = imread('tree-stump-texture.jpg');

% %% extract features
% feats = GetDenseFeature(img, {'color_Lab', 'texture_LM'}, struct('verbosity', 1));

%% extract texton features
feats = GetDenseFeature(img, {'texton'}, struct('nTexton', 64, 'verbosity', 2));