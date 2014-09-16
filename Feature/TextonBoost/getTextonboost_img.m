function [ o_feats, o_valid ] = getTextonboost_img( i_tbParams, i_textonizedImg, i_nTextons )
%GETTEXTONBOOST Summary of this function goes here
%   Detailed explanation goes here

parts = i_tbParams.parts;
nParts = i_tbParams.nParts;
nTextons = i_nTextons;
WH = i_tbParams.WH;
WH_half = round(WH/2);
imgWH = [size(i_textonizedImg, 2); size(i_textonizedImg, 1)];
nSkip = 16;

[Xs, Ys] = meshgrid(1:nSkip:size(i_textonizedImg, 2), 1:nSkip:size(i_textonizedImg, 1));
Xs = Xs(:);
Ys = Ys(:);
nPix = numel(Xs);

o_feats = zeros(nParts*nTextons, nPix);
o_valid = false(1, nPix);
for pxInd=1:nPix
    curXY = [Xs(pxInd); Ys(pxInd)];
    if any(curXY - (WH_half + 1) < 1) || any(curXY + (WH_half + 1) > imgWH)
        continue;
    end
    
    xy_tl = curXY-WH_half;
    
    fInd=1;
    for pInd=1:nParts
        curPart = parts(:, pInd);
        xy_part_tl = [curPart(1); curPart(3)] + xy_tl - 1;
        xy_part_br = [curPart(2); curPart(4)] + xy_tl - 1;
        partArea = prod(xy_part_br - xy_part_tl);
        
        for tInd=1:nTextons
            curTexImg = i_textonizedImg == tInd;
            [pxs, pys] = meshgrid(xy_part_tl(1):xy_part_br(1), xy_part_tl(2):xy_part_br(2));
            o_feats(fInd, pxInd) = sum(sum(curTexImg(pys, pxs)))/partArea;
            o_valid(1, pxInd) = true;
            
            fInd = fInd + 1;
        end
    end
end

end

