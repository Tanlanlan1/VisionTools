% %% inputss
% srcImgs = '~/UPenn/Dropbox/img';
% annDir = srcImgs;
% imgExt = 'jpg';

function annotateBB(srcDir, annDir, imgFmt)

% for efficiency
imgList = dir(sprintf('%s/%s', srcDir, imgFmt));

% pop-up the figure
h = figure('Name', 'Annotator', 'CreateFcn', @(src, evt) initFig(src, evt, srcDir, annDir, imgList), 'KeyPressFcn', @(obj, evt) handleKeyEvent(obj, evt, srcDir, annDir, imgList));
    
% wait
uiwait(h);
    
end

function showImg(img, name)
clf;
imagesc(img);
title(sprintf('fn: %s ([f]inish/[n]ext/[p]rev/ne[w]/[s]ave)', name), 'Interpreter', 'none');
end

function showAnns(anns, imSz)
global hRects;
for aInd=1:numel(anns)
    hRects = [hRects; imrect(gca, anns(aInd).xywh)]; % [xmn ymin width height]
    textPos = anns(aInd).xywh(1:2);
    textPos(2) = imSz(1)-textPos(2);
%     textPos(2) = textPos(2) + 0.05;
    annotation('textbox', [textPos./imSz([2, 1]) 0.0 0.0], 'String', anns(aInd).label, 'Color', 'r');
end
end

function initFig(src, evt, srcDir, annDir, imgList)
global iInd;
global aInd;
global anns;

iInd = 1;

[~, fid, ~] = fileparts(imgList(iInd).name);
annfn = sprintf('%s/%s_bb.mat', annDir, fid);
if exist(annfn, 'file')
    load(annfn);
else
    anns = struct('label', {}, 'xywh', {});
end
aInd = 1;

% show the img
img = imread(sprintf('%s/%s', srcDir, imgList(iInd).name));
showImg(img, imgList(iInd).name);
% show anns
showAnns(anns, size(img)); 
end

function handleKeyEvent(obj, evt, srcImgDir, annDir, imgList)
global iInd;
global aInd;
global anns;
global hRects;

switch evt.Key
    case 'f' % finish annotation
        close(obj);
    case 'n' % show next image
        iInd = min(iInd+1, numel(imgList));
        
        [~, fid, ~] = fileparts(imgList(iInd).name);
        annfn = sprintf('%s/%s_bb.mat', annDir, fid);
        if exist(annfn, 'file')
            load(annfn);
        else
            anns = struct('label', {}, 'xywh', {});
        end
        aInd = 1;
        hRects = [];
        
        % show the img
        img = imread(sprintf('%s/%s', srcImgDir, imgList(iInd).name));
        showImg(img, imgList(iInd).name);
        % show anns
        showAnns(anns, size(img)); 
        
    case 'p' % show previous image
        
        iInd = max(iInd-1, 1);
        
        [~, fid, ~] = fileparts(imgList(iInd).name);
        annfn = sprintf('%s/%s_bb.mat', annDir, fid);
        if exist(annfn, 'file')
            load(annfn);
        else
            anns = struct('label', {}, 'xywh', {});
        end
        aInd = 1;
        hRects = [];
        
        % show the img
        img = imread(sprintf('%s/%s', srcImgDir, imgList(iInd).name));
        showImg(img, imgList(iInd).name);
        % show anns
        showAnns(anns, size(img)); 
        
    case 's' % save annotations
        
        [~, fid, ~] = fileparts(imgList(iInd).name);
        annfn = sprintf('%s/%s_bb.mat', annDir, fid);
        save(annfn, 'anns');
                
    case 'w' % draw new annotation
        
        % draw
        hRect = imrect; % [xmn ymin width height]
        hRects = [hRects hRect];
        
        % get the label name
        curLabel = input('input label > ', 's');
        
        % save
        anns(aInd).label = curLabel;
        anns(aInd).xywh = getPosition(hRect);
        aInd = aInd + 1;
end
end
