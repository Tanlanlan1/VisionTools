%% init
totTic = tic;
addpath(genpath('../Feature')); %%FIXME
addpath(genpath('../JointBoost')); %%FIXME
addpath(genpath('../libMatlabHelper')); %%FIXME
close all;

annotate = true;
saveAnnotation = true;
verbosity = 1;
nPerClsSample = 200;
resizeRatio = 1;


if annotate
    trainInd = 1;
    testInd = 2;
else
    clsList = 83;
    trainInd = 1;
    testInd = 2;
    nCls = numel(clsList)+1;
end


% TextonBoost params
TBParams = struct(...
    'samplingRatio', 0.1, ...
    'nTexton', 64, ...
    'nPart', 16, ...
    'LOFilterWH', [201; 201], ...
    'verbosity', verbosity);

% JointBoost paramsimg1
JBParams = struct(...
    'nWeakLearner', 200, ... 
    'featDim', TBParams.nPart*TBParams.nTexton, ...
    'featSelRatio', 0.1, ...
    'featValRange', 0:0.1:1, ...
    'verbosity', verbosity);

if annotate
    % obtain annotations
    img1 = imresize(imread('ted1.jpg'), resizeRatio);
    img2 = imresize(imread('ted2.jpg'), resizeRatio);
    if exist('ann.mat', 'file') && saveAnnotation
        load('ann.mat');
    else
        rects = [];
        figure(100); 
        title('Enter for exit');
        imshow(img1);
        while true
            figure(100); 
            [x, y] = ginput(2);
            if isempty(x) && isempty(y)
                break;
            end
            rect = [min(x) min(y) max(x)-min(x)+1 max(y)-min(y)+1];
            hold on;
            rectangle('Position', rect, 'EdgeColor', 'r', 'LineWidth', 5);
            hold off;
            pause(0.1);
            rects = [rects; rect];
        end
        save('ann.mat', 'rects');
    end
    % construct data structure
    imgs = struct('img', {img1, img2});
    labels = struct('cls', {zeros(size(img1, 1), size(img1, 2)), []});
    for rInd=1:size(rects, 1)
        rect = rects(rInd, :);
        mask = poly2mask(...
            [rect(1) rect(1) rect(1)+rect(3)-1 rect(1)+rect(3)-1], ...
            [rect(2) rect(2)+rect(4)-1 rect(2)+rect(4)-1 rect(2)], ...
            size(img1, 1), size(img1, 2));
        labels(1).cls(mask) = rInd;
    end
    JBParams.nCls = size(rects, 1) + 1; % count bg %%FIXME: bg should be label 0?
    labels(1).cls(labels(1).cls == 0) = JBParams.nCls;
else
    % load db
    DBSt = load('DB/nyu_depth_v2_sample.mat');
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
    JBParams.nCls = nCls;
end

%% learn
[mdls, params] = LearnSemSeg(imgs(trainInd), labels(trainInd), struct('pad', true, 'feat', TBParams, 'nPerClsSample', nPerClsSample,'classifier', JBParams, 'verbosity', 1));

%% predict
[cls, vals, params] = PredSemSeg(imgs(testInd), mdls, params);

%% show
showPred( cls, vals, params, imgs(trainInd).img, labels(trainInd), imgs(testInd).img );

%%
fprintf('* Total time: %s\n', num2str(toc(totTic)));