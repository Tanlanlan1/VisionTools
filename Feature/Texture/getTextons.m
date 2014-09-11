function [ o_textons ] = getTextons( i_trainImgs, i_sampleStep, i_scales, i_K, i_fb )
%GETTEXTON Summary of this function goes here
%   Detailed explanation goes here

nTrImg = size(i_trainImgs, 4);
nScale = numel(i_scales);
%%
trainImgs_cell = cell(nTrImg, nScale);
parfor i=1:nTrImg
    curImgs = cell(1, nScale);
    curImg = im2double(rgb2gray(i_trainImgs(:, :, :, i)));
    tmppy = buildImgPyramid(i_scales, curImg, []);
    for s=1:nScale
        curImgs{s} = tmppy(s).img;
    end
    trainImgs_cell(i, :) = curImgs;
end

%% get filter responses
disp('* getting filter responses...');
data_cell = cell(size(trainImgs_cell, 1), size(trainImgs_cell, 2));
for imgInd=1:size(trainImgs_cell, 1)
    fprintf('- %d/%d\n', imgInd, size(trainImgs_cell, 1));
    sampleStep = i_sampleStep*i_scales;
    data_cell_s = cell(1, size(trainImgs_cell, 2));
    for sInd=1:size(trainImgs_cell, 2)
        curImg = trainImgs_cell{imgInd, sInd};
        curResp = zeros(size(curImg, 1), size(curImg, 2), size(i_fb, 3));
        for fInd=1:size(i_fb, 3)
            curResp(:, :, fInd) = conv2(curImg, i_fb(:, :, fInd), 'same');
        end
        data_cell_s{1, sInd} = reshape(curResp, [size(curResp, 1)*size(curResp, 2) size(curResp, 3)]);
        data_cell_s{1, sInd} = data_cell_s{1, sInd}(1:round(sampleStep(sInd)):end, :);
    end
    data_cell(imgInd, :) = data_cell_s;
end

%% find textons
data = cell2mat(data_cell(:));
fprintf('* clustering %d data...\n', size(data, 1));
o_textons = vl_kmeans(data', i_K, 'Algorithm', 'Elkan');

%% visualize
fb_c = {};
for i=1:size(i_fb, 3)
    fb_c{i} = i_fb(:, :, i);
end
[tim,tperm] = visTextons(o_textons, fb_c);
figure(20001); clf;
imgarray = cell2mat(reshape({tim{tperm}}, [1 1 1 i_K]));
montage(cell2mat(reshape({tim{tperm}}, [1 1 1 i_K])), 'DisplayRange', [min(imgarray(:)), max(imgarray(:))])
axis image; colorbar;


end

