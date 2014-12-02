%% init
totTic = tic;
addpath(genpath('../Feature')); %%FIXME
addpath(genpath('../Superpixel')); %%FIXME
addpath(genpath('../JointBoost')); %%FIXME
addpath(genpath('../libMatlabHelper')); %%FIXME
close all;
if matlabpool('size') == 0
    matlabpool open;
end

annotate = true;
saveAnnotation = true;
verbosity = 1;
nPerClsSample = 500;
resizeRatio = 1;
scales = 0.5:0.25:2;

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
    'nTexton', 64, ... % 64
    'nPart', 100, ... % 100
    'LOFilterWH', [101; 101], ...%%FIXME: adoptive??
    'verbosity', verbosity);

% JointBoost params
JBParams = struct(...
    'nWeakLearner', 200, ...
    'binary', 1, ...
    'learnBG', 0, ...
    'featDim', TBParams.nPart*TBParams.nTexton, ...
    'featSelRatio', 0.1, ...
    'featValRange', 0:0.1:1, ...
    'verbosity', verbosity);

if annotate
    % read images
    img1 = imresize(imread('ted1_d5.jpg'), resizeRatio);
    img2 = imresize(imread('ted2_d5.jpg'), resizeRatio);
    % obtain annotations
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
    nCls = size(rects, 1) + 1; % count bg as a new class
    % duplicate images and construct scale space for training
    imgs_train = struct('img', [], 'scale', [], 'pivot', []);
    nW_train = 2;
    nH_train = 2;
    for iInd1=1:nH_train
        for iInd2=1:nW_train
            for sInd=1:numel(scales)
                imgs_train(iInd1, iInd2, sInd).img = imresize(img1, scales(sInd));
                imgs_train(iInd1, iInd2, sInd).scale = scales(sInd);
                imgs_train(iInd1, iInd2, sInd).pivot = scales(sInd) == 1;
            end
        end
    end
    % select only target scale
    imgs_train = imgs_train(:, :, scales == 1);
    % duplicate images and construct scale space for test
    imgs_test = struct('img', [], 'scale', [], 'pivot', []);
    nW = 1;
    nH = 1;
    for iInd1=1:nH
        for iInd2=1:nW
            for sInd=1:numel(scales)
                imgs_test(iInd1, iInd2, sInd).img = imresize(img2, scales(sInd));
                imgs_test(iInd1, iInd2, sInd).scale = scales(sInd);
                imgs_test(iInd1, iInd2, sInd).pivot = scales(sInd) == 1;
            end
        end
    end
    % construct label structure for training
    labels = struct('cls', [], 'depth', []);
    for iInd=1:numel(imgs_train)
        curImg = imgs_train(iInd).img;
        labels(iInd).cls = zeros(size(curImg, 1), size(curImg, 2), nCls);
        if imgs_train(iInd).scale == 1 % select only target scale
            for rInd=1:size(rects, 1)
                rect = rects(rInd, :);
                mask = poly2mask(...
                    [rect(1) rect(1) rect(1)+rect(3)-1 rect(1)+rect(3)-1], ...
                    [rect(2) rect(2)+rect(4)-1 rect(2)+rect(4)-1 rect(2)], ...
                    size(curImg, 1), size(curImg, 2));
                labels(iInd).cls(:, :, rInd) = mask;
            end
        end
        labels(iInd).cls(:, :, nCls) = ~sum(labels(iInd).cls(:, :, 1:nCls-1), 3);    
    end
    labels = reshape(labels, size(imgs_train));
    % set mdlRects
    mdlRects = struct('iInd', struct('poly', []));
    for oInd=1:size(rects, 1) 
        curRect = rects(oInd, :);
        for iInd=1:numel(imgs_train)
            xy = [[curRect(1) curRect(1) curRect(1)+curRect(3)-1 curRect(1)+curRect(3)-1];
                [curRect(2) curRect(2)+curRect(4)-1 curRect(2)+curRect(4)-1 curRect(2)]];
            K = convhull(xy(1, :), xy(2, :));
            curPoly = xy(:, K); %same for all images
            mdlRects(oInd).iInd(iInd).poly = curPoly;
        end
    end
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
end

JBParams.nCls = nCls;
SemSegParams = struct(...
    'pad', true, ...
    'feat', TBParams, ...
    'nPerClsSample', nPerClsSample, ...
    'classifier', JBParams, ...
    'mdlRects', mdlRects,...
    'scales', scales, ...
    'supPix', 1, ...
    'verbosity', 1);

%% learn
[mdls, params_learn] = LearnSemSeg(imgs_train, labels, SemSegParams);

%% predict
assert(any(SemSegParams.scales == 1));
[pred, params_pred] = PredSemSeg(imgs_test, mdls, params_learn);

%% show
showPred( pred, params_pred, imgs_train, labels, imgs_test );

%% end
fprintf('* Total time: %s\n', num2str(toc(totTic)));