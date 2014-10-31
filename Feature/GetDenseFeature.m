function [ o_feat, o_params ] = GetDenseFeature( i_imgs, i_cues, i_params )
% 
%   Matlab wrapper of dense feature extract methods
%   
% ----------
%   Input: 
% 
%       i_imgs          a struct array of images
%       i_cues:         a string array where each elements stands for cues, e.g. Lab, texture
%           'Color_RGB'         extract RGB colors
%           'Color_Lab'         extract Lab colors
%           'Texture_LM'        extract texture based on the the Leung-Malik filter bank
%           'Texton'            extract textons based on the texture_LM
%           'TextonBoost'       extract textonBoost
%           'TextonBoostInt'
% 
%       i_params:       parameter structure for each cues
%           i_params.verbosity      the level of verbosity [0 (slient), 1(console output), 2(+selected figures), 3(all)]  
%           i_params.samplingRatio  
%           i_params.nNN
%           i_params.nTexton        ('texton', 'textonBoost') the number of textons
%           i_params.nPart          ('textonBoost') the number of a subwindow
%           i_params.LOFilterWH     ('textonBoost') the width and height of a
%                                   layout filter
%           i_params.sampleMask(rm)     ('textonBoost' or for all) extract features only
%                                   on the selected region (FIXME: Not used, use samplingRatio)
% 
% ----------
%   Output:
% 
%       o_feat:        a struct array of dense features
% 
% ----------
%   DEPENDENCY:
%   
%cmp
% ----------
% Written by Sangdon Park (sangdonp@cis.upenn.edu), 2014.
% All rights reserved.
%

%% init
thisFilePath = fileparts(mfilename('fullpath'));
addpath([thisFilePath '/Texture']);
addpath([thisFilePath '/TextonBoost']);
vlfeatmexpath = [thisFilePath '/../vlfeat/toolbox/mex'];
vlfeatmexapthall = genpath(vlfeatmexpath);
addpath(vlfeatmexapthall);
if ~strfind(getenv('LD_LIBRARY_PATH'), vlfeatmexapthall)
    setenv('LD_LIBRARY_PATH', [vlfeatmexapthall ':' getenv('LD_LIBRARY_PATH')]);
end

if nargin < 3
    i_params = struct('verbosity', 0);
end
if ~isfield(i_params, 'verbosity')
    i_params.verbosity = 0;
end

%%FIXME: samplingRatio, samplingMask...duplicated concepts, samplingMask is
%%not used now


%% extract features
for iInd=1:numel(i_imgs)
    i_imgs(iInd).img = im2double(i_imgs(iInd).img);
end
imgs = i_imgs;

% o_feat = cell(1, 1, numel(i_cues));
% o_feat = struct('feat', [], 'cue', []);
assert(numel(i_cues) == 1);
for cInd=1:numel(i_cues)
    switch i_cues{cInd}
        case 'Color_RGB'
            assert(size(imgs(1).img, 3) == 3);
            o_feat = GetRGBDenseFeature(imgs);
            
        case 'Color_Lab'
            assert(size(imgs, 3) == 3);    
            assert(~iscell(imgs));
            o_feat = GetLabDenseFeature(imgs);
            
        case 'Texture_LM'
            assert(~iscell(imgs));
            o_feat = GetTextureLMFeature(imgs);
             
        case 'Texture_MR8'
%             img = rgb2gray(img);
            o_feat = GetTextureMR8Feature(imgs);
            
        case 'Texton'
%             [o_feat{cInd}, o_params] = GetTextonFeature(imgs, i_params);
            [o_feat, o_params] = GetTextonFeature(imgs, i_params);
            
        case 'TextonBoost'
%             [o_feat{cInd}, o_params] = GetTextonBoostFeature(imgs, i_params);
            [o_feat, o_params] = GetTextonBoostFeature(imgs, i_params);
            
        case 'TextonBoostInt'
%             [o_feat{cInd}, o_params] = GetTextonBoostIntFeature(imgs, i_params);
            [o_feat, o_params] = GetTextonBoostIntFeature(imgs, i_params);
            
        otherwise
            warning('Wrong cue name: %s', i_cues{cInd});
    end
    
end

%% return
% o_feat = cell2mat(cellfun(@(x) cell2mat(x), o_feat, 'UniformOutput', false));
% o_feat = cell2mat(o_feat);

%% show
if i_params.verbosity >= 3
    iInd = randi(numel(o_feat));
    fInd = randi(size(o_feat(iInd).feat, 3));
    
    h = figure(23912);
    figure(h);
    imagesc(o_feat(iInd).feat(:, :, fInd));
    axis image;
    colorbar;
    title(sprintf('%dth image, %dth feature', iInd, fInd));

end

end

function [o_feat] = GetRGBDenseFeature(i_img)
o_feat = i_img;
end

function [o_feat] = GetLabDenseFeature(i_img)
o_feat = applycform(i_img, makecform('srgb2lab'));
end

function [o_feat, o_filterBank] = GetTextureLMFeature(i_img)

nImgs = numel(i_img);
Fs = makeLMfilters;
o_feat = i_img;
for i=1:nImgs
    img = rgb2gray(i_img(i).img);
    img_pad = padarray(img, [(size(Fs, 1)-1)/2 (size(Fs, 2)-1)/2], 'symmetric', 'both');
    responses = zeros(size(img, 1), size(img, 2), size(Fs, 3));
    for fInd=1:size(Fs, 3)
        responses(:, :, fInd) = conv2(img_pad, Fs(:, :, fInd), 'valid'); % symetric filters, so don't need to flip
    end
    o_feat(i).Texture = responses;
end
o_filterBank = Fs;


% img = rgb2gray(i_img);
% Fs = makeLMfilters;
% img_pad = padarray(img, [(size(Fs, 1)-1)/2 (size(Fs, 2)-1)/2], 'symmetric', 'both');
% responses = zeros(size(img, 1), size(img, 2), size(Fs, 3));
% for fInd=1:size(Fs, 3)
%     responses(:, :, fInd) = conv2(img_pad, Fs(:, :, fInd), 'valid'); % symetric filters, so don't need to flip
% end
% o_feat = responses;
% o_filterBank = Fs;
end

function [o_feat] = GetTextureMR8Feature(i_img)

nImgs = numel(i_img);
FSz = 51;
o_feat = i_img;
for i=1:nImgs
    img = im2double(rgb2gray(i_img(i).img));
    img_pad = padarray(img, [(FSz-1)/2 (FSz-1)/2], 'symmetric', 'both');
    responses = MR8fast(img_pad);
    responses = (reshape(shiftdim(responses, 1), [size(img, 1) size(img, 2) 8]));
    
    o_feat(i).Texture = responses;
end

end

function [o_feats, o_params] = GetTextonFeature(i_imgs, i_params)
%% check i_params
assert(isfield(i_params, 'nTexton'));
nTexton = i_params.nTexton;

if ~isfield(i_params, 'samplingRatio')
    i_params.samplingRatio = 1;
end
samplingRatio = i_params.samplingRatio;

if ~isfield(i_params, 'nNN')
    i_params.nNN = max(1, round(i_params.nTexton*0.1));
end
nNN = i_params.nNN;

verbosity = i_params.verbosity;


% if iscell(i_imgs)
%     nImg = numel(i_imgs);
%     cellFlag = true;
% else
%     i_imgs = {i_imgs};
%     nImg = 1;
%     cellFlag = false;
% end
nImgs = numel(i_imgs);

% %% construct image pyramid
% if verbosity >= 1
%     disp('* construct an image pyramid');
% end
% img_py = cell(nImg, nScale);
% for iInd=1:nImg
%     for sInd=1:nScale
%         img_py{iInd, sInd} = imresize(i_imgs{iInd}, scales(sInd));
%     end
% end

feats = i_imgs;
%% obtain texture
if verbosity >= 1
    disp('* obtain texture information');
end

if ~isfield(i_imgs, 'Texture')
    [feats, fb] = GetTextureLMFeature(feats);
%     feats = GetTextureMR8Feature(feats);
    for iInd=1:nImgs %%FIXME: not good...
        % add colors
        feats(iInd).Texture = cat(3, feats(iInd).Texture, GetRGBDenseFeature(feats(iInd).img));
    end
end

%% extract textons if not exist
if ~isfield(i_params, 'textons') || isempty(i_params.textons)
    if verbosity >= 1
        fprintf('* extract textons');
    end
    data = cell(1, nImgs);
    for iInd=1:nImgs
        curTexture = feats(iInd).Texture;
        data_is = reshape(curTexture, [size(curTexture, 1)*size(curTexture, 2) size(curTexture, 3)])';
        step = round(1/samplingRatio);
        data{iInd} = data_is(:, 1:step:end);
    end
    data = cell2mat(data);
    if verbosity >= 1
        fprintf(' from %d data\n', size(data, 2));
    end
    textons = vl_kmeans(data, nTexton, 'Algorithm', 'Elkan');
    kdtree = vl_kdtreebuild(textons); % L2 distance
else
    textons = i_params.textons;
    kdtree = i_params.kdtree;
end

%% obtain texton features
% feats = struct('feat', []);
% feats(nImgs) = feats;

if ~isfield(i_imgs, 'Texton')
    for iInd=1:nImgs
        curTexture = feats(iInd).Texture;
        curTexture_q = reshape(curTexture, [size(curTexture, 1)*size(curTexture, 2) size(curTexture, 3)])';

        [IND, DIST] = vl_kdtreequery(kdtree, textons, curTexture_q, 'numNeighbors', nNN);
        textonImg = zeros(size(curTexture, 1), size(curTexture, 2), nTexton);
        for nnInd=1:nNN
            [cols, rows] = meshgrid(1:size(textonImg, 2), 1:size(textonImg, 1));
            linInd = sub2ind(size(textonImg), rows(:), cols(:), double(IND(nnInd, :))');
            textonImg(linInd) = exp(-DIST(nnInd, :));
        end
        feats(iInd).Texton = textonImg;
    end
end

%% visualize
if verbosity >= 2
    % textons
    
    fb_c = cell(size(fb, 3), 1);
    for i=1:size(fb, 3)
        fb_c{i} = fb(:, :, i);
    end
    [tim, tperm] = visTextons(textons, fb_c);
    resp_norm = cellfun(@(r) 0.01*r./max(abs(r(:))), tim(tperm), 'UniformOutput', false);
    imgarray = cell2mat(reshape(resp_norm, [1 1 1 nTexton]));
    figure(71683); clf;
    montage(imgarray, 'DisplayRange', [min(imgarray(:)), max(imgarray(:))]);
    axis image; colorbar;

    % texton feature
    riInd = randi(nImgs, 1);
    curImg = feats(riInd).img;
    [~ , curFeat_max] = max(feats(riInd).feat, [], 3);
    figure(76234); clf;
    subplot(1, 2, 1); imshow(curImg);
    subplot(1, 2, 2); imagesc(curFeat_max); axis image;
end

%% return
% if ~cellFlag
%     feat = cell2mat(feat);
% end
o_feats = feats;
o_params = i_params;
o_params.textons = textons;
o_params.kdtree = kdtree;


end

function [ o_parts ] = GenTextonBoostParts( i_params )

LOFWH = i_params.LOFilterWH;
nParts = i_params.nPart;
verbosity = i_params.verbosity;

o_parts = [];
for pInd=1:nParts
    xs = [0; 0];
    while xs(1) == xs(2)
        xs = randi(LOFWH(1), [2, 1]);
    end
    
    ys = [0; 0];
    while ys(1) == ys(2)
        ys = randi(LOFWH(2), [2, 1]);
    end
    
    o_parts = [o_parts [min(xs); max(xs); min(ys); max(ys)]];
end

if verbosity >= 2
    figure(34125); clf;
    rectangle('Position', [1 1 LOFWH(:)'-1]); hold on; % layoutFilter
    for pInd=1:nParts
        subWin = o_parts(:, pInd)';
        rectangle('Position', [subWin(1) subWin(3) subWin([2,4]) - subWin([1 3])]); hold on; % subwindows
    end
end

end

function [o_feats, o_params] = GetTextonBoostIntFeature(i_imgs, i_params)

%% init
assert(isfield(i_params, 'nPart'));
assert(isfield(i_params, 'nTexton'));
assert(isfield(i_params, 'LOFilterWH'));

% if iscell(i_imgs)
%     nImg = numel(i_imgs);
%     cellFlag = true;
% else
%     i_imgs = {i_imgs};
%     nImg = 1;
%     cellFlag = false;
% end
nImg = numel(i_imgs);

if mod(i_params.LOFilterWH(1), 1) ~= 0
    if i_params.verbosity >= 1
        warning('non-integer width of a layout filter. Will be floored');
    end
    i_params.LOFilterWH(1) = floor(i_params.LOFilterWH(1));
end
if mod(i_params.LOFilterWH(2), 1) ~= 0
    if i_params.verbosity >= 1
        warning('non-integer height of a layout filter. Will be floored');
    end
    i_params.LOFilterWH(2) = floor(i_params.LOFilterWH(2));
end

if mod(i_params.LOFilterWH(1), 2) == 0
    if i_params.verbosity >= 1
        warning('even width of a layout filter. Will be modified to be odd');
    end
    i_params.LOFilterWH(1) = i_params.LOFilterWH(1) - 1;
end
if mod(i_params.LOFilterWH(2), 2) == 0
    if i_params.verbosity >= 1
        warning('even height of a layout filter. Will be modified to be odd');
    end
    i_params.LOFilterWH(2) = i_params.LOFilterWH(2) - 1;
end

params = i_params;
nTexton = i_params.nTexton;

%% generate parts
if ~isfield(params, 'parts')
    params.parts = GenTextonBoostParts(params);
end

%% obtain Texton features
textonFeats = i_imgs;
if ~isfield(textonFeats, 'Texton')
    [textonFeats, params] = GetTextonFeature(i_imgs, params);
end

%% obtain integral images

% feat = cell(nImg, 1);
% feats = struct('feat', []);
% feats(nImg) = feats;
feats = textonFeats;
if ~isfield(feats, 'TextonIntImg')
    for iInd=1:nImg
        curFeat = textonFeats(iInd).Texton;
        textIntImg = zeros(size(curFeat, 1)+1, size(curFeat, 2)+1, nTexton, 'single');
        for tInd=1:nTexton
            textIntImg(:, :, tInd) = integralImage(curFeat(:, :, tInd));
        end
        feats(iInd).TextonIntImg = textIntImg;
    end
end

%% return
% if ~cellFlag
%     feat = cell2mat(feat);
% end
o_params = params;
o_feats = feats;

end

function [o_feat, o_params] = GetTextonBoostFeature(i_imgs, i_params)
%% init
verbosity = i_params.verbosity;

%% obtain integral images
[textIntImgs, params] = GetTextonBoostIntFeature(i_imgs, i_params);

% if iscell(textIntImgs)
%     cellFlag = true;
%     textIntImgs = cell2mat(reshape(textIntImgs, 1, 1, 1, numel(textIntImgs)));
%     
%     waring('still working...');
%     keyboard;
% else
%     cellFlag = false;
% end
%% extract TextonBoost features
[feat, params] = GetTextonBoost(textIntImgs, params);

%% return
% if cellFlag 
%    feat = squeeze(mat2cell(feat, size(feat, 1), size(feat, 2), size(feat, 3), ones(1, size(feat, 4))));
% end    
o_params = params;
o_feat = feat;

%% visualize
if verbosity >= 2
    figure(56457); 
    imagesc(o_feat(:, :, randi(size(o_feat, 3), 1), randi(size(o_feat, 4), 1)));
end
end

