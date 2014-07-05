function showDetectionResults
%% user settings
objName = 'window';
load(sprintf('/data/v50/sangdonp/FacadeRecognition/data/Paris/pascalFormat_2/learnObjectUsingDPM5/2010/%s_final.mat', objName));
baseDir = '/data/v50/sangdonp/FacadeRecognition/data/Paris/pascalFormat_2/VOC2010/VOCdevkit/Paris/';
imgDir = [baseDir 'JPEGImages/'];
imgsetPath = [baseDir 'ImageSets/Main/test.txt'];
resultDir = '/data/v50/sangdonp/FacadeRecognition/results/detections/';
range = [-5, 5];
thres = [0 5; -1 0; -2 -1];

%% show
figure(100);
cm = colormap(jet(59));
    
ids = textread(imgsetPath, '%s');
for tInd=1:size(thres, 1)
    curThres = thres(tInd, :);
    curResultDir = sprintf('%sscoresfrom%dto%d', resultDir, curThres(1), curThres(2));
    try
        rmdir(curResultDir, 's');
    catch
    end
    mkdir(curResultDir);

    for i=1:numel(ids);
        cla;

        fn = sprintf('%s%s.jpg', imgDir, ids{i});
        im = imread(fn);

        % detect
        bbox = process(im, model, range(1));

        % show
        showColorBoxes(im, bbox, range, cm, curThres);

        % save       
        print(gcf, '-djpeg', sprintf('%s/%s_%s.jpg', curResultDir, ids{i}, objName));

    end
end
end

function showColorBoxes(im, boxes, range, cm, curThres)
% Draw bounding boxes on top of an image.
%   showboxes(im, boxes, out)
%
%   If out is given, a pdf of the image is generated (requires export_fig).

nCM = size(cm, 1);
cwidth = 2;

image(im); 
% axis image;
% axis off;
% set(gcf, 'Color', 'white');

if ~isempty(boxes)
    for bInd=size(boxes, 1):-1:1
  
        x1 = boxes(bInd,1);
        y1 = boxes(bInd,2);
        x2 = boxes(bInd,3);
        y2 = boxes(bInd,4);
        score = boxes(bInd,6);
        if score < curThres(1) || score > curThres(2)
            continue;
        end
        
        % remove unused filters
        del = find(((x1 == 0) .* (x2 == 0) .* (y1 == 0) .* (y2 == 0)) == 1);
        x1(del) = [];
        x2(del) = [];
        y1(del) = [];
        y2(del) = [];
        
        colorInd = round((score-range(1))/(range(2) - range(1))*(nCM-1) + 1);
        c = cm(colorInd, :);
        s = '-';
        
        line([x1 x1 x2 x2 x1]', [y1 y2 y2 y1 y1]', 'color', c, 'linewidth', cwidth, 'linestyle', s);
    end

end

cb_h = colorbar;
set(cb_h, 'YTick', [1 10:10:nCM]);
set(cb_h, 'YTickLabel', [range(1):(range(2)-range(1))/6:range(2)]);
end