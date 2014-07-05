function [ o_pasDB ] = convBB2Pascal( i_im, i_imFN, i_imDN, i_DBN, i_bbs, i_objNames )
%CONV2VOC Summary of this function goes here
%   bbs(:, i): [xmin; ymin; xmax; ymax]
%   objNames{i}: an object name of bbs(:, i)

%% convert bbs in the Pascal format (pascal 2009)
curImgFileNm = i_imFN;
curImg = i_im;
imgSize = size(curImg);
bbs = i_bbs;
objNames = i_objNames;


PasDBAnno = [];
PasDBAnno.folder = i_imDN;
PasDBAnno.filename = curImgFileNm;
PasDBAnno.source.database = i_DBN;
PasDBAnno.source.annotation = i_DBN;
PasDBAnno.source.image = i_DBN;
PasDBAnno.source.scene = i_DBN;

PasDBAnno.size.width = imgSize(2);
PasDBAnno.size.height = imgSize(1);
if numel(imgSize) == 2
    PasDBAnno.size.depth = 1;
else
    PasDBAnno.size.depth = imgSize(3);
end

PasDBAnno.segmented = '0';
PasDBAnno.objects = [];


curBBs = bbs;
for bbInd=1:size(curBBs, 2)

    obj.class = objNames{bbInd};
    obj.occluded = 0;
    obj.truncated = 0;
    obj.difficult = 0;
    obj.pose = 'Unspecified';
    obj.bndbox.xmax = curBBs(3, bbInd);
    obj.bndbox.xmin = curBBs(1, bbInd);
    obj.bndbox.ymax = curBBs(4, bbInd);
    obj.bndbox.ymin = curBBs(2, bbInd);
    obj.bbox = curBBs(:, bbInd)';

    PasDBAnno.objects = [PasDBAnno.objects obj];
end
curPasDB = [];
curPasDB.annotation = PasDBAnno;

%% return
o_pasDB = curPasDB;

end

