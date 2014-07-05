%% inputss
srcImgs = '~/UPenn/Dropbox/img';
annDir = srcImgs;
imgExt = 'jpg';


%% 
hFig = 52323;
imgList = dir(sprintf('%s/*.%s', srcImgs, imgExt));
for iInd=1:numel(imgList)
    [~, fid, ~] = fileparts(imgList(iInd).name);
    annfn = sprintf('%s/%s_bb.mat', annDir, fid);
    
    figure(hFig); clf;
    imagesc(imread(sprintf('%s/%s', srcImgs, imgList(iInd).name)));
    title(sprintf('%s (save: double click, discard: close window)', imgList(iInd).name));
    rect = [];
    if exist(annfn, 'file')
        load(annfn);
        rect = wait(imrect(gca, rect)); % [xmn ymin width height]
    else
        rect = wait(imrect); % [xmn ymin width height]
    end
    if isempty(rect)
        continue;
    end
    save(annfn, 'rect');
end
