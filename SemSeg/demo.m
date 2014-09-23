%% init
addpath(genpath('../Feature')); %%FIXME
addpath(genpath('../JointBoost')); %%FIXME

% TextonBoost params
TBParams = struct(...
    'samplingRatio', 0.1, ...
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

%% learn
[mdls, params] = LearnSemSeg(imgs(trainInd), labels(trainInd), struct('feat', TBParams, 'classifier', JBParams));

%% predict
[cls, vals, params] = PredSemSeg(imgs(testInd), mdls, params);

%% show
showPred( cls, vals, params, imgs(trainInd).img, labels(trainInd), imgs(testInd).img );
