%% init
addpath(genpath('../Feature')); %%FIXME
addpath(genpath('../JointBoost')); %%FIXME

annotate = true;

if annotate
    trainInd = 1;
    testInd = 2;
else
    clsList = 83;
    trainInd = 1;
    testInd = 2;
end


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

if annotate
    % obtain annotations
    img1 = imread('ted1.jpg');
    img2 = imread('ted2.jpg');
    rects = [];
    figure(1); 
    title('Enter for exit');
    imshow(img1);
    while true
        figure(1); 
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

%% learn
[mdls, params] = LearnSemSeg(imgs(trainInd), labels(trainInd), struct('pad', true, 'feat', TBParams, 'classifier', JBParams));

%% predict
[cls, vals, params] = PredSemSeg(imgs(testInd), mdls, params);

%% show
showPred( cls, vals, params, imgs(trainInd).img, labels(trainInd), imgs(testInd).img );
