%% init
addpath(genpath('../Feature')); %%FIXME
addpath(genpath('../JointBoost')); %%FIXME

% load db
DBSt = load('DB/nyu_depth_v2_sample.mat');
clsList = 83;
trainInd = 1;
testInd = 2;
% build input arguments
imgs = struct('img', []);
for iInd=1:size(DBSt.images, 4)
    imgs(iInd).img = DBSt.images(:, :, :, iInd);
end

labels = struct('cls', [], 'depth', []);
for lInd=1:size(labels, 3)
    labels(lInd).cls = (numel(clsList)+1)*ones(size(DBSt.labels(:, :, lInd)));
    for cInd=1:numel(clsList)
        ind = DBSt.labels(:, :, lInd) == clsList(cInd);
        labels(lInd).cls(ind) = cInd;
    end
    labels(lInd).depth = DBSt.depths(:, :, lInd);
end

% TextonBoost params
TBParams = struct(...
    'samplingRatio', 0.05, ...
    'nTexton', 64, ...
    'nPart', 16, ...
    'LOFilterWH', [101; 101], ...
    'verbosity', 1);

% JointBoost params
JBParams = struct(...
    'nCls', numel(clsList)+1, ...
    'nWeakLearner', 500, ...
    'featDim', TBParams.nPart*TBParams.nTexton, ...
    'featSelRatio', 0.1, ...
    'featValRange', 0:0.1:1, ...
    'verbosity', 1);

% SemSeg params

%% learn
[mdls, params] = LearnSemSeg(imgs(trainInd), labels(trainInd), struct('feat', TBParams, 'classifier', JBParams));

%% predict
[cls, vals, params] = PredSemSeg(imgs(testInd), mdls, params);

%% show
sampleMask = params.feat.sampleMask;
[rows, cols] = find(sampleMask);

figure(1); clf;
subplot(1, 2, 1);
trImg = imgs(trainInd).img;
imshow(trImg);
% rectangle('Position', rect, 'EdgeColor', 'r');
% title('training regions represented by the bounding box');
hold on;
heatMap = labels(trainInd).cls;
heatMap(heatMap ~= 1) = 0;
heatMap(heatMap == 1) = 255;
h = imagesc(heatMap);
set(h, 'AlphaData', 0.7);
hold off;
title('training regions represented by the bounding box');

teImg = imgs(testInd).img;
subplot(1, 2, 2);
imshow(teImg);
hold on;
heatMap = zeros(size(sampleMask));
heatMap(sampleMask) = reshape(cls, [numel(unique(rows)) numel(unique(cols))]);
heatMap(heatMap ~= 1) = 0;
heatMap(heatMap == 1) = 255;
h = imagesc(heatMap);
set(h, 'AlphaData', 0.7);
hold off;
title('Identified regions by the trained classifier');


