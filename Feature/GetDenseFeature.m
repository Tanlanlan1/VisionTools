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
%           i_params.nParts          ('textonBoost') the number of a subwindow
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
run([thisFilePath '/../vlfeat/toolbox/vl_setup.m']);
% vlfeatmexpath = [thisFilePath '/../vlfeat/toolbox'];
% vlfeatmexapthall = genpath(vlfeatmexpath);
% addpath(vlfeatmexapthall);
% vlfeatmexpath = [thisFilePath '/../vlfeat/toolbox/mex/mexa64'];
% if isempty(strfind(getenv('LD_LIBRARY_PATH'), vlfeatmexpath))
%     setenv('LD_LIBRARY_PATH', [vlfeatmexpath ':' getenv('LD_LIBRARY_PATH')]);
% end

if nargin < 3
    i_params = struct('verbosity', 0);
end
if ~isfield(i_params, 'verbosity')
    i_params.verbosity = 0;
end
o_params = i_params;

%% extract features
for iInd=1:numel(i_imgs)
    i_imgs(iInd).img = im2double(i_imgs(iInd).img);
end
imgs = i_imgs;

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
        
        case 'Texture'
            assert(~iscell(imgs));
            o_feat = GetTextureLMFeature(imgs, i_params);
            o_params = i_params;
            
        case 'Texture_LM'
            assert(~iscell(imgs));
            o_feat = GetTextureLMFeature(imgs, i_params);
            o_params = i_params;
             
        case 'Texture_MR8'
%             img = rgb2gray(img);
            o_feat = GetTextureMR8Feature(imgs);
            
        case 'SelfSimilarity'
            o_feat = GetSelfSimilarityFeature( imgs, i_params );
            
        case 'TextonInit'
            o_feat = [];
            [o_params] = GetTexton(imgs, i_params);
            
        case 'VW'
            [o_feat, o_params] = GetVisualWordFeature(i_imgs, i_params);
            
        case 'Texton'
%             [o_feat{cInd}, o_params] = GetTextonFeature(imgs, i_params);
            [o_feat, o_params] = GetTextonFeature(imgs, i_params);

        case 'Img2TextonEff'
            [o_feat, o_params] = GetTextonFeatureEff(imgs, i_params);
            
        case 'TextonBoostInit'
            o_feat = [];
            o_params = InitTextonBoostParts(i_params);
            
        case 'TextonBoost'
%             [o_feat{cInd}, o_params] = GetTextonBoostFeature(imgs, i_params);
            [o_feat, o_params] = GetTextonBoostFeature(imgs, i_params);
            
        case 'TextonBoostInt'
%             [o_feat{cInd}, o_params] = GetTextonBoostIntFeature(imgs, i_params);
            [o_feat, o_params] = GetTextonBoostIntFeature(imgs, i_params);
            
        case 'DSIFT'
            o_feat = GetDenseSIFTFeature(imgs, i_params);
        otherwise
            warning('Wrong cue name: %s', i_cues{cInd});
    end
    
end

%% return
% o_feat = cell2mat(cellfun(@(x) cell2mat(x), o_feat, 'UniformOutput', false));
% o_feat = cell2mat(o_feat);

% %% show
% if i_params.verbosity >= 3
%     iInd = randi(numel(o_feat));
%     fInd = randi(size(o_feat(iInd).feat, 3));
%     
%     h = figure(23912);
%     figure(h);
%     imagesc(o_feat(iInd).feat(:, :, fInd));
%     axis image;
%     colorbar;
%     title(sprintf('%dth image, %dth feature', iInd, fInd));
% 
% end

end

function SaveData(fn, data)
save(fn, 'data');
end

function [data] = LoadData(fn)
ldst = load(fn);
data = ldst.data;
end

function [ o_feats ] = GetSelfSimilarityFeature( i_imgs, i_params )
%GETSELFSIMILARITYFEATURE Summary of this function goes here
%   Detailed explanation goes here

warning('SelfSimilarity: default parameter settings');
ssparams = struct('patch_size', 5, 'desc_rad', 40, 'nrad', 3, 'nang', 12, 'var_noise', 300000, 'saliency_thresh', 0.7, 'homogeneity_thresh', 0.7, 'snn_thresh', 0.85);

o_feats = i_imgs;
o_feats(1).SelfSimilarity = [];
%% compute features
parfor iInd=1:numel(i_imgs)
    % pad
    ssdmargin = ssparams.desc_rad + (ssparams.patch_size-1)/2;
    I = i_imgs(iInd).img;
    assert(isa(I, 'double') && all(I(:)<2));
    I_gray = padarray(rgb2gray(I), [ssdmargin ssdmargin], 'symmetric', 'both');
    I_ssd = double(im2uint8(I_gray));
    % fn
    [pathstr, name, ~] = fileparts(i_imgs(iInd).filename);
    fn = sprintf('%s/%s_scale_%s_selfsimilarity.mat', pathstr, name, num2str(i_imgs(iInd).scale, 4));
    if i_params.fSaveFeats
        if exist(fn, 'file')
            % load
            ssdesc = LoadData(fn);
        else
            % compute
            ssdesc = ComputeSelfSimilarityFeature(I_ssd, ssparams);
            SaveData(fn, ssdesc);
        end
    else
        ssdesc = ComputeSelfSimilarityFeature(I_ssd, ssparams);
    end
    % reshape
    feats = zeros(size(i_imgs(iInd).img, 1), size(i_imgs(iInd).img, 2), size(ssdesc.resp, 1));
    linind = sub2ind([size(I_gray, 1), size(I_gray, 2)], ssdesc.draw_coords(2, :), ssdesc.draw_coords(1, :));
    for fInd=1:size(feats, 3);
        feat = zeros(size(feats, 1)+ssdmargin*2, size(feats, 2)+ssdmargin*2);
        feat(linind) = ssdesc.resp(fInd, :);
        % discard results on the pad
        feat(1:ssdmargin, :) = [];
        feat(end-ssdmargin+1:end, :) = [];
        feat(:, 1:ssdmargin) = [];
        feat(:, end-ssdmargin+1:end) = [];
        % save
        feats(:, :, fInd) = feat;
    end
    o_feats(iInd).SelfSimilarity = feats;
end

end

function [ssdesc] = ComputeSelfSimilarityFeature(I, ssparams)
% assert(isa(I, 'double') && all(I(:)<2));
tID = tic;
% [resp, draw_coords, salient_coords, homogeneous_coords, snn_coords] = mexCalcSsdescs(double(im2uint8(rgb2gray(I))), ssparams);
[resp, draw_coords, salient_coords, homogeneous_coords, snn_coords] = mexCalcSsdescs(I, ssparams);
fprintf('* selfsimilarity...%s sec.\n', num2str(toc(tID)));
ssdesc = struct('resp', resp, 'draw_coords', draw_coords, 'salient_coords', salient_coords, 'homogeneous_coords', homogeneous_coords, 'snn_coords', snn_coords);
end


function [o_feat] = GetDenseSIFTFeature(i_imgs, i_params)
%% init
o_feat = i_imgs;
nImgs = numel(i_imgs);
verbosity = i_params.verbosity;
o_feat(1).DSIFT = [];
%% extract
if verbosity >= 1
    sTic = tic;
    fprintf('[DSIFT] Obtain DSIFT...');
end
parfor iInd=1:nImgs
    % fn
    [pathstr, name, ~] = fileparts(o_feat(iInd).filename);
    fn = sprintf('%s/%s_scale_%s_DSIFT.mat', pathstr, name, num2str(o_feat(iInd).scale, 4));
    
    if i_params.fSaveFeats && exist(fn, 'file')
        responses = LoadData(fn);
    else
        % compute
        I = im2single(rgb2gray(i_imgs(iInd).img));
        [f, d] = vl_dsift(I);
        % reshape??
        assert(f(2, 1) < f(2, 2));
        d = shiftdim(d, 1);
        d = reshape(d, size(I, 1)-9, size(I, 2)-9, size(d, 2)); %%FIXME: const!
        denseFeat = padarray(d, [4, 4], 0, 'pre'); %%FIXME: const!
        responses = padarray(denseFeat, [5 5], 0, 'post'); %%FIXME: const!
        % save
        if i_params.fSaveFeats
            SaveData(fn, responses);
        end

    end
    o_feat(iInd).DSIFT = responses;
end
%% fin
if verbosity >= 1
    fprintf('done (%s sec.)\n', num2str(toc(sTic)));
end
end

function [o_feat] = GetRGBDenseFeature(i_img)
o_feat = i_img;
end

function [o_feat] = GetLabDenseFeature(i_img)
o_feat = applycform(i_img, makecform('srgb2lab'));
end

function [o_feat, o_filterBank] = GetTextureLMFeature(i_img, i_params)
nImgs = numel(i_img);
Fs = makeLMfilters;
o_feat = i_img;
o_filterBank = Fs;
verbosity = i_params.verbosity;
% check precomputed results
if isfield(i_img, 'Texture')
    warning('use precomputed Texture in the structure')
    return;
end
if verbosity >= 1
    sTic = tic;
    fprintf('[Texture] Obtain texture responses...');
end
%% compute

for i=1:nImgs
    % fn
    [pathstr, name, ~] = fileparts(o_feat(i).filename);
    fn = sprintf('%s/%s_scale_%s_LMTextureFilters.mat', pathstr, name, num2str(o_feat(i).scale, 4));
    
    if i_params.fSaveFeats && exist(fn, 'file')
        if verbosity >= 1
            fprintf('load...');
        end
        responses = LoadData(fn);
    else
        % pad images
        img = rgb2gray(i_img(i).img);
        img_pad = padarray(img, [(size(Fs, 1)-1)/2 (size(Fs, 2)-1)/2], 'symmetric', 'both');
        filters = rot90(Fs, 2);
        % compute
        responses = zeros(size(img, 1), size(img, 2), size(Fs, 3));
        for fInd=1:size(filters, 3)
            responses(:, :, fInd) = conv2(img_pad, filters(:, :, fInd), 'valid'); % symetric filters, so don't need to flip
        end
        % save
        if i_params.fSaveFeats
            SaveData(fn, responses);
        end
    end
    o_feat(i).Texture = responses;


%     % pad images
%     img = rgb2gray(i_img(i).img);
%     img_pad = padarray(img, [(size(Fs, 1)-1)/2 (size(Fs, 2)-1)/2], 'symmetric', 'both');
%     filters = rot90(Fs, 2);
%     % compute
%     responses = zeros(size(img, 1), size(img, 2), size(Fs, 3));
%     for fInd=1:size(filters, 3)
%         responses(:, :, fInd) = conv2(img_pad, filters(:, :, fInd), 'valid'); % symetric filters, so don't need to flip
%     end
%     o_feat(i).Texture = responses;
end

if verbosity >= 1
    fprintf('done (%s sec.)\n', num2str(toc(sTic)));
end
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

function [o_params] = GetTexton(i_imgs, i_params)
%% check i_params
assert(isfield(i_params, 'nTexton'));
nTexton = i_params.nTexton;

% if ~isfield(i_params, 'samplingRatio')
%     i_params.samplingRatio = 1;
% end
% samplingRatio = i_params.samplingRatio;
assert(isfield(i_params, 'nSamplesForVW'));
nSamplesForVW = i_params.nSamplesForVW;

verbosity = i_params.verbosity;
nImgs = numel(i_imgs);
feats = i_imgs;

%% extract textons if not exist
if ~isfield(i_params, 'textons') || isempty(i_params.textons)
    % get texture
    if ~isfield(i_imgs, 'Texture')
        [feats, fb] = GetTextureLMFeature(feats, i_params);
    end
    % get texton features
    if verbosity >= 1
        sTic = tic;
        fprintf('[Texton] Find textons...');
    end
    data = cell(1, nImgs);
    for iInd=1:nImgs
        curTexture = feats(iInd).Texture;
        data_is = reshape(curTexture, [size(curTexture, 1)*size(curTexture, 2) size(curTexture, 3)])';
%         step = round(1/samplingRatio);
%         data{iInd} = data_is(:, 1:step:end);
        ind_samples = randi(size(data_is, 2), [1 nSamplesForVW]);
        data{iInd} = data_is(:, ind_samples);
    end
    data = cell2mat(data);
    if verbosity >= 1
        fprintf('from %d data...', size(data, 2));
    end
    textons = vl_kmeans(data, nTexton, 'Algorithm', 'Elkan');
    kdtree = vl_kdtreebuild(textons); % L2 distance

    if verbosity >= 1
        fprintf('done (%s sec.)\n', num2str(toc(sTic)));
    end 
    if verbosity >= 3 
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
    end
else
    textons = i_params.textons;
    kdtree = i_params.kdtree;
end

%% return
o_params = i_params;
o_params.textons = textons;
o_params.kdtree = kdtree;


end

function [o_params] = GetVisualWord(i_imgs, i_cues, i_params)
%% check i_params
assert(isfield(i_params, 'nTexton'));
nTexton = i_params.nTexton;

% if ~isfield(i_params, 'samplingRatio')
%     i_params.samplingRatio = 1;
% end
samplingRatio = i_params.samplingRatio;
assert(isfield(i_params, 'nSamplesForVW'));
nSamplesForVW = i_params.nSamplesForVW;

verbosity = i_params.verbosity;
nImgs = numel(i_imgs);
feats = i_imgs;
o_params = i_params;
%% extract textons if not exist
for cInd=1:numel(i_cues)
    cueStr = i_cues{cInd};
    if ~isfield(i_params, 'textons') || isempty(i_params.textons)
        % get texton features
        if verbosity >= 1
            sTic = tic;
            fprintf('[VisualWord] Find visual words for %s...', cueStr);
        end
        % load if available
        [pathstr, ~, ~] = fileparts(feats(1).filename);
        fn = sprintf('%s/VWs_%s_Cues_%s.mat', pathstr, i_params.VW.IDStr, cueStr);
        if i_params.fSaveFeats && exist(fn, 'file')
            if verbosity >= 1
                fprintf('load...');
            end
            data = LoadData(fn);
            VWs = data.VWs;
            kdtree = data.kdtree;
        else

            % get basic features
            assert(isfield(i_imgs, cueStr));
%             switch(cueStr)
%                 case 'Texture'
%                     assert(isfield(i_imgs, 'Texture'));
%     %                 if ~isfield(i_imgs, 'Texture')
%     %                     [feats, fb] = GetTextureLMFeature(feats, i_params);
%     %                 end
%                 case 'DSIFT'
%                     assert(isfield(i_imgs, 'DSIFT'));
%     %                 if ~isfield(i_imgs, 'DSIFT')
%     %                     feats = GetDenseSIFTFeature(feats);
%     %                 end
%             end

            % get data
            data = cell(1, nImgs);
            for iInd=1:nImgs
                curTexture = getfield(feats, {iInd}, cueStr);
                data_is = reshape(curTexture, [size(curTexture, 1)*size(curTexture, 2) size(curTexture, 3)])';
                step = round(1/samplingRatio);
                data{iInd} = data_is(:, 1:step:end);
    %             ind_samples = randi(size(data_is, 2), [1 nSamplesForVW]);
    %             data{iInd} = data_is(:, ind_samples);
            end
            data = double(cell2mat(data));
            if verbosity >= 1
                fprintf('from %d data...', size(data, 2));
            end
            % find VWs
            nTextonsEach = nTexton/numel(i_cues);
            VWs = vl_kmeans(data, nTextonsEach, 'Algorithm', 'Elkan', 'Distance', i_params.VW.distance); 
            kdtree = vl_kdtreebuild(VWs, 'Distance', i_params.VW.distance); 

            % save
            SaveData(fn, struct('VWs', VWs, 'kdtree', kdtree));
        end
        
        o_params.textons{cInd} = VWs;
        o_params.kdtree{cInd} = kdtree;
        
        
%         dimension = size(data, 1);
%         numClusters = nTextonsEach;
%         [initMeans, assignments] = vl_kmeans(data, nTextonsEach, 'Algorithm', 'Elkan', 'Distance', 'L1');
%         initCovariances = zeros(dimension,numClusters);
%         initPriors = zeros(1,numClusters);
% 
%         % Find the initial means, covariances and priors
%         for i=1:numClusters
%             data_k = data(:,assignments==i);
%             initPriors(i) = size(data_k,2) / numClusters;
% 
%             if size(data_k,1) == 0 || size(data_k,2) == 0
%                 initCovariances(:,i) = diag(cov(data'));
%             else
%                 initCovariances(:,i) = diag(cov(data_k'));
%             end
%         end
%         
%         o_params.textons{cInd} = vl_gmm(data, nTextonsEach, ...
%             'initialization','custom', ...
%             'InitMeans',initMeans, ...
%             'InitCovariances',initCovariances, ...
%             'InitPriors',initPriors);
%         o_params.kdtree{cInd} = vl_kdtreebuild(o_params.textons{cInd}); % L2 distance

        if verbosity >= 1
            fprintf('done (%s sec.)\n', num2str(toc(sTic)));
        end
        if verbosity >= 3
            if strcmp(cueStr, 'Texture')
                fb = makeLMfilters;
                % visualize vws
                vws = o_params.textons{cInd};
                vws_filters = cell(1, 1, 1, nTextonsEach);
                for vwInd=1:nTextonsEach
                    vws_filters{vwInd} = sum(bsxfun(@times, fb, reshape(vws(:, vwInd), [1 1 numel(vws(:, vwInd))])), 3);
                    vws_filters{vwInd} = vws_filters{vwInd}./norm(vws_filters{vwInd}(:)); 
                end

                figure(1451243); 
                I_vws = cell2mat(vws_filters);
                montage(I_vws, 'DisplayRange', [min(I_vws(:)) max(I_vws(:))]);
            end
        end

    end
end
%% return

end

function [o_feats, o_params] = GetVisualWordFeature(i_imgs, i_params)

% if ~isfield(i_params, 'nNN')
%     i_params.nNN = max(1, round(i_params.nTexton*0.5));
% end
assert(isfield(i_params.VW, 'nNN'));
assert(isfield(i_params.VW, 'sig'));
baseFeats = i_params.baseFeats;
nNN = i_params.VW.nNN;
sig = i_params.VW.sig;
verbosity = i_params.verbosity;
nImgs = numel(i_imgs);
feats = i_imgs;
nTexton = i_params.nTexton;

%% extract textons
i_params = GetVisualWord(i_imgs, baseFeats, i_params);
textons = i_params.textons;
kdtree = i_params.kdtree;

%% obtain texton features
if ~isfield(i_imgs, 'Texton')
    if verbosity >= 1
        sTic = tic;
        fprintf('[VisualWord] Obtain visual word responses...');
    end
    [pathstr, ~, ~] = fileparts(feats(1).filename);
    figDir = sprintf('%s/VWResponses', pathstr);
    feats(1).Texton = [];
%     feats_added = cell(numel(feats), 1);
    if sig == 0
        % check statistics of distnaces
        dist_sum = 0;
        nData = 0;
        for iInd=1:nImgs
            % extract
            for cInd=1:numel(baseFeats)
                curTexture = getfield(feats(iInd), baseFeats{cInd});
                curTexture_q = double(reshape(curTexture, [size(curTexture, 1)*size(curTexture, 2) size(curTexture, 3)])');

                [~, DIST] = vl_kdtreequery(kdtree{cInd}, textons{cInd}, curTexture_q, 'numNeighbors', nNN);
                dist_sum = dist_sum + sum(DIST(:));
                nData = nData + numel(DIST);
            end
        end
        d_ave = dist_sum/nData;
        sig = -d_ave/log(0.5);
    end
    
    % extract vw responses
    for iInd=1:nImgs
        % fn
        [pathstr, name, ~] = fileparts(feats(iInd).filename);
        fn = sprintf('%s/%s_scale_%s_VW_%s_Cues_%s_Resp.mat', pathstr, name, num2str(feats(iInd).scale, 4), i_params.VW.IDStr, cell2mat(baseFeats));
        
        if i_params.fSaveFeats && exist(fn, 'file')
            if verbosity >= 1
                fprintf('load...');
            end
            TextonSt = LoadData(fn);
        else
            % prepare for save figures
            if i_params.fSaveFeats && iInd == 1
%                 [~, ~, ~] = rmdir(figDir, 's');
                [~, ~, ~] = mkdir(figDir);
            end
            % extract
            curFeat = feats(iInd);
            textonImgs = cell(1, 1, numel(baseFeats));
            nTextonsEach = nTexton/numel(baseFeats);
            for cInd=1:numel(baseFeats)
                curTexture = getfield(curFeat, baseFeats{cInd});
                curTexture_q = double(reshape(curTexture, [size(curTexture, 1)*size(curTexture, 2) size(curTexture, 3)])');

                [IND, DIST] = vl_kdtreequery(kdtree{cInd}, textons{cInd}, curTexture_q, 'numNeighbors', nNN);
                textonImg = zeros(size(curTexture, 1), size(curTexture, 2), nTextonsEach);
                [cols, rows] = meshgrid(1:size(textonImg, 2), 1:size(textonImg, 1)); %%FIXME: inefficient
                for nnInd=1:nNN
                    linInd = sub2ind(size(textonImg), rows(:), cols(:), double(IND(nnInd, :))');
                    textonImg(linInd) = exp(-DIST(nnInd, :)/sig);
                end
    %             textonImgs{1, 1, cInd} = bsxfun(@times, textonImg, 1./sum(abs(textonImg), 3)); %L1 normalization
                textonImgs{1, 1, cInd} = textonImg;
            end
            textonImg = cell2mat(textonImgs);
            assert(size(textonImg, 3) == nTexton);
            % sparsify
            TextonSt = struct('textonSpImg', []);
            for tInd=1:nTexton
                TextonSt(tInd).textonSpImg = sparse(textonImg(:, :, tInd));
            end
            % save
            if i_params.fSaveFeats
                SaveData(fn, TextonSt);
            end
            % save
            if i_params.fSaveFeats && mod(iInd, round(nImgs*0.1))==0
                % save VWs responses
                h = 6248;
                for i=1:round(numel(TextonSt)*0.1):numel(TextonSt)
                    sfigure(h); clf; 
                    imagesc(curFeat.img);
                    hold on;
                    imagesc(TextonSt(i).textonSpImg); 
                    caxis([0 1]);
                    colorbar;
                    alpha(0.7);
                    hold off;
                    saveas(h, sprintf('%s/%s_scale_%s_Cues_%s_VWID_%s_VW_%.5d_Resp.png', figDir, name, num2str(feats(iInd).scale, 4), cell2mat(baseFeats), i_params.VW.IDStr, i));
                end
            end
        end
%         curFeat.Texton = TextonSt;
%         feats_added{iInd} = curFeat;
        feats(iInd).Texton = TextonSt;
    end
    %%

    if i_params.fSaveFeats
%         % VW response range
%         nVWs = numel(feats(1).Texton);
%         for vwInd=1:nVWs
%             vwhist_cell = cell(numel(feats), 1);
%             for iInd=1:numel(feats)
%                 vwhist_cell{iInd} = feats(iInd).Texton(vwInd).textonSpImg(:);
%             end
%             vwhist_fn = sprintf('%s/%s_scale_%s_VW_%.5d_VWRespHist.png', figDir, name, num2str(feats(iInd).scale, 4), vwInd);
%             if ~exist(vwhist_fn, 'file')
%                 h = 1423;
%                 figure(h); clf;
%                 vwhist = cell2mat(vwhist_cell);
%                 vwhist(vwhist == 0) = []; % rm zeros
%                 hist(vwhist, 0:0.01:1);
%                 title(sprintf('Histogram of %dth VW response values (except for zeros)', vwInd));
%                 saveas(h, vwhist_fn);
%             end
%         end
        
        % average VWs
    end

    if verbosity >= 1
        fprintf('done (%s sec.)\n', num2str(toc(sTic)));
    end
end

%% decide the range of VWs response values
assert(isfield(i_params, 'nFeatThres'));
if ~isfield(i_params, 'featThres')
    nVWs = i_params.nTexton;
    nFeatThres = i_params.nFeatThres;
    % fn
    [pathstr, ~, ~] = fileparts(feats(1).filename);
    fn = sprintf('%s/featThres_VWID_%s_nVW_%d_nFeatThres_%d.mat', pathstr, i_params.VW.IDStr, nVWs, nFeatThres);
    if ~exist(fn, 'file')
        featThres = zeros(nFeatThres, nVWs);
        for vwInd=1:nVWs
            % calc hist.
            vwhist_cell = cell(numel(feats), 1);
            for iInd=1:numel(feats)
                vwhist_cell{iInd} = feats(iInd).Texton(vwInd).textonSpImg(:);
            end
            vwhist = cell2mat(vwhist_cell);
            vwhist(vwhist == 0) = []; % rm zeros
            vwhist = vwhist(:);
            % find the uniform range
            val_min = min(vwhist);
            val_max = max(vwhist);
            thres = val_min:(val_max-val_min)/(nFeatThres+1):val_max;
            featThres(:, vwInd) = thres(2:end-1);
            % save
            if i_params.fSaveFeats && mod(vwInd, 10) == 0
                vwhist_fn = sprintf('%s/%s_scale_%s_VW_%.5d_VWRespHist.png', figDir, name, num2str(feats(iInd).scale, 4), vwInd);
                h = 1423;
                sfigure(h); clf;
                hist(vwhist, 0:0.01:1);
                title(sprintf('Histogram of %dth VW response values (except for zeros)', vwInd));
                saveas(h, vwhist_fn);
            end
            % save
            save(fn, 'featThres');
        end
    else
        load(fn);
    end
    i_params.featThres = featThres;
end

%% return
o_feats = feats;
o_params = i_params;

end

function [o_feats, o_params] = GetTextonFeature(i_imgs, i_params)
% %% check i_params
if ~isfield(i_params, 'nNN')
    i_params.nNN = max(1, round(i_params.nTexton*0.1));
end
nNN = i_params.nNN;
verbosity = i_params.verbosity;
nImgs = numel(i_imgs);
feats = i_imgs;
nTexton = i_params.nTexton;
%% extract textons
i_params = GetTexton(i_imgs, i_params);
textons = i_params.textons;
kdtree = i_params.kdtree;
    
%% obtain texture
if ~isfield(i_imgs, 'Texture')
    [feats, fb] = GetTextureLMFeature(feats, i_params);
end

%% obtain texton features
if ~isfield(i_imgs, 'Texton')
    if verbosity >= 1
        sTic = tic;
        fprintf('[Texton] Obtain texton responses...');
    end
    feats_added = cell(numel(feats), 1);
    for iInd=1:nImgs
        curFeat = feats(iInd);
        curTexture = curFeat.Texture;
        curTexture_q = reshape(curTexture, [size(curTexture, 1)*size(curTexture, 2) size(curTexture, 3)])';

        [IND, DIST] = vl_kdtreequery(kdtree, textons, curTexture_q, 'numNeighbors', nNN);
        textonImg = zeros(size(curTexture, 1), size(curTexture, 2), nTexton);
        for nnInd=1:nNN
            [cols, rows] = meshgrid(1:size(textonImg, 2), 1:size(textonImg, 1));
            linInd = sub2ind(size(textonImg), rows(:), cols(:), double(IND(nnInd, :))');
            textonImg(linInd) = exp(-DIST(nnInd, :));
        end
%         curFeat.Texton = textonImg;
%         feats_added{iInd} = curFeat;
        % sparsify
        TextonSt = struct('textonSpImg', []);
        for tInd=1:nTexton
            TextonSt(tInd).textonSpImg = sparse(textonImg(:, :, tInd));
        end
        curFeat.Texton = TextonSt;
        feats_added{iInd} = curFeat;
    end
    feats = reshape(cell2mat(feats_added), size(i_imgs));
    
    if verbosity >= 1
        fprintf('done (%s sec.)\n', num2str(toc(sTic)));
    end
    %% visualize
    if verbosity >= 3
    %     % textons
    %     fb_c = cell(size(fb, 3), 1);
    %     for i=1:size(fb, 3)
    %         fb_c{i} = fb(:, :, i);
    %     end
    %     [tim, tperm] = visTextons(textons, fb_c);
    %     resp_norm = cellfun(@(r) 0.01*r./max(abs(r(:))), tim(tperm), 'UniformOutput', false);
    %     imgarray = cell2mat(reshape(resp_norm, [1 1 1 nTexton]));
    %     figure(71683); clf;
    %     montage(imgarray, 'DisplayRange', [min(imgarray(:)), max(imgarray(:))]);
    %     axis image; colorbar;

        % texton feature
        riInd = randi(nImgs, 1);
        curImg = feats(riInd).img;
        [~ , curFeat_max] = max(feats(riInd).feat, [], 3);
        figure(76234); clf;
        subplot(1, 2, 1); imshow(curImg);
        subplot(1, 2, 2); imagesc(curFeat_max); axis image;
    end
    
end




%% return
o_feats = feats;
o_params = i_params;
% o_params.textons = textons;
% o_params.kdtree = kdtree;


end

function [o_feats, o_params] = GetTextonFeatureEff(i_imgs, i_params)

%% extract textons

if ~isfield(i_params, 'nNN')
    i_params.nNN = max(1, round(i_params.nTexton*0.1));
end
nNN = i_params.nNN;
verbosity = i_params.verbosity;
nImgs = numel(i_imgs);
feats = i_imgs;
nTexton = i_params.nTexton;
%% extract textons
i_params = GetTexton(i_imgs, i_params);
textons = i_params.textons;
kdtree = i_params.kdtree;
    
%% obtain texton features
if ~isfield(i_imgs, 'Texton')
    if verbosity >= 1
        sTic = tic;
        fprintf('[Texton] Extract texton features...');
    end
    feats_added = cell(numel(feats), 1);
    for iInd=1:nImgs
        curFeat = feats(iInd);
        curTexture = curFeat.Texture;
        curTexture_q = reshape(curTexture, [size(curTexture, 1)*size(curTexture, 2) size(curTexture, 3)])';

        [IND, DIST] = vl_kdtreequery(kdtree, textons, curTexture_q, 'numNeighbors', nNN);
        textonImg = zeros(size(curTexture, 1), size(curTexture, 2), nTexton);
        for nnInd=1:nNN
            [cols, rows] = meshgrid(1:size(textonImg, 2), 1:size(textonImg, 1));
            linInd = sub2ind(size(textonImg), rows(:), cols(:), double(IND(nnInd, :))');
            textonImg(linInd) = exp(-DIST(nnInd, :));
        end
%         curFeat.Texton = textonImg;
%         feats_added{iInd} = curFeat;
        % sparsify
        TextonSt = struct('textonSpImg', []);
        for tInd=1:nTexton
            TextonSt(tInd).textonSpImg = sparse(textonImg(:, :, tInd));
        end
        curFeat.Texton = TextonSt;
        feats_added{iInd} = curFeat;
    end
    feats = reshape(cell2mat(feats_added), size(i_imgs));
    
    if verbosity >= 1
        fprintf('done (%s sec.)\n', num2str(toc(sTic)));
    end
end


%% visualize
if verbosity >= 3
    % texton feature
    riInd = randi(nImgs, 1);
    curImg = feats(riInd).img;
    [~ , curFeat_max] = max(feats(riInd).feat, [], 3);
    figure(76234); clf;
    subplot(1, 2, 1); imshow(curImg);
    subplot(1, 2, 2); imagesc(curFeat_max); axis image;
end

%% return
o_feats = feats;
o_params = i_params;
% o_params.textons = textons;
% o_params.kdtree = kdtree;


end


function [ o_parts ] = GenTextonBoostParts( i_params )

LOFWH = i_params.LOFilterWH;
nParts = i_params.nParts;
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
    
    o_parts = [o_parts [min(xs); max(xs); min(ys); max(ys)]]; %%FIXME: how about [x y w h]??
end

if verbosity >= 3
    figure(34125); clf;
    rectangle('Position', [1 1 LOFWH(:)'-1]); hold on; % layoutFilter
    for pInd=1:nParts
        subWin = o_parts(:, pInd)';
        rectangle('Position', [subWin(1) subWin(3) subWin([2,4]) - subWin([1 3])]); hold on; % subwindows
    end
end

end

function [o_params] = InitTextonBoostParts(i_params)
assert(isfield(i_params, 'nParts'));
assert(isfield(i_params, 'LOFilterWH'));
verbosity = i_params.verbosity;
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

%% init parts
if verbosity >= 1
    sTic = tic;
    fprintf('[Textonboost subwindows] Randomly generate subwindows: ');
end

if ~isfield(i_params, 'parts')
    % fn
    fn = sprintf('%s/subwindows_Textonboost.mat', i_params.cacheDir);
    if ~exist(fn, 'file')
        if verbosity >= 1
            fprintf('generate...');
        end
        parts = GenTextonBoostParts(i_params);
        save(fn, 'parts');
    else
        if verbosity >= 1
            fprintf('load from cache...');
        end 
        load(fn, 'parts');
    end
    i_params.parts = parts;
    
else
    if verbosity >= 1
        fprintf('use the existing one...');
    end
end
if verbosity >= 1
    fprintf('done (%s sec.)\n', num2str(toc(sTic)));
end

%% return
o_params = i_params;


end

function [o_feats, o_params] = GetTextonBoostIntFeature(i_imgs, i_params)

%% init
assert(isfield(i_params, 'nParts'));
assert(isfield(i_params, 'nTexton'));
assert(isfield(i_params, 'LOFilterWH'));
verbosity = i_params.verbosity;
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


assert(isfield(params, 'parts'));
% %% generate parts
% if ~isfield(params, 'parts')
%     params.parts = GenTextonBoostParts(params);
% end

%% obtain Texton features
textonFeats = i_imgs;
assert(isfield(textonFeats, 'Texton'));
% if ~isfield(textonFeats, 'Texton')
%     [textonFeats, params] = GetTextonFeature(i_imgs, params);
% end

%% obtain integral images
if verbosity >= 1
    sTic = tic;
    fprintf('[Textonboost] Obtain integral images...');
end
feats = textonFeats;
if ~isfield(feats, 'TextonIntImg')
    intImgDataType = 'single';
    % obtain integral images for each images
    for iInd=1:nImg
        % load cache results if there is
        [pathstr, name, ~] = fileparts(feats(iInd).filename);
        fn = sprintf('%s/%s_scale_%s_VW_%s_Cues_%s_IntetralImage.mat', pathstr, name, num2str(feats(iInd).scale, 4), i_params.VW.IDStr, cell2mat(i_params.baseFeats));
        if i_params.fSaveFeats && exist(fn, 'file')
            if verbosity >= 1
                fprintf('load...');
            end
            load(fn);
        else
            curFeat = textonFeats(iInd).Texton;
            textIntImg = zeros(size(curFeat(1).textonSpImg, 1)+1, size(curFeat(1).textonSpImg, 2)+1, nTexton, intImgDataType);
            % obtain an integral image for each vw
            for tInd=1:nTexton
                textIntImg(:, :, tInd) = integralImage(full(curFeat(tInd).textonSpImg));
            end
            % save
            if i_params.fSaveFeats
                save(fn, 'textIntImg');
            end
        end
        feats(iInd).TextonIntImg = textIntImg;
    end
    
    % compression
    if i_params.fCompression 
        if verbosity >= 1
            fprintf('compress data...');
        end
        % check the range of integral image values
        nBits = 8; % FIXME: compress only into uint8
        decodeMap = zeros(2^nBits, params.nTexton, intImgDataType);
        for tInd=1:nTexton
            for sInd=1:size(feats, 3)
                feat_scale = feats(:, :, sInd);
                vwvs = cell(size(feat_scale, 1)*size(feat_scale, 2), 1);
%                 feat_scale = feats;
%                 vwvs = cell(numel(feat_scale), 1);
                for iInd=1:numel(vwvs)
                    vwvs{iInd} = feat_scale(iInd).TextonIntImg(:, :, tInd);
                    vwvs{iInd} = vwvs{iInd}(:);
                    vwvs{iInd} = vwvs{iInd}./max(vwvs{iInd}); % normalization to make value range [0 1]
                end
                vwvs = double(cell2mat(vwvs));
                val_min = min(vwvs);
                val_max = max(vwvs);
                nBin = 2^nBits;
                rng = val_min:(val_max-val_min)/nBin:val_max;
                assert(numel(rng) == nBin+1);


                figure(43124);
                vwvs(vwvs == 0) = [];
                hist(vwvs, rng); title(sprintf('Histogram of integral image values in %d th VW', tInd));
            end
        end
        
        % save decodeMap
    
    end
        
    
end
if verbosity >= 1
    fprintf('done (%s sec.)\n', num2str(toc(sTic)));
end
%% return
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

