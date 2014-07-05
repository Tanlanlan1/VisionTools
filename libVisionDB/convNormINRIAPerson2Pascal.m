%% user settings
vockitDir = '../libPascal';
mhDir = '../libMatlabHelper';

ojCls = 'inriaperson';

dbName = 'VOC_INRIA_person';
newDBName = 'VOC_INRIA_person';
% srcDBDir = ['/data/v50/sangdonp/objectDetection/DB/' dbName];
srcImgDir = '/data/v50/sangdonp/objectDetection/DB/INRIA_person/INRIAPerson/train_64x128_H96/pos/';
srcAnnDir = [];
srcNegImgDir = '/data/v50/sangdonp/objectDetection/DB/INRIA_person/INRIAPerson/train_64x128_H96/neg/';

destDBDir = ['/data/v50/sangdonp/objectDetection/DB/' newDBName];
destAnnDir = [destDBDir '/Annotations/'];
destImgDir = [destDBDir '/JPEGImages/'];
imgSetsDir = [destDBDir '/ImageSets/Main/'];

padding = false;
trainingRatio = 0.25;
validateRatio = 0.25;
testRatio = 0.5;
preferImgArea = 200*100;
bgImgArea = 400*400;

minNegBBWH = [2, 2];

genRandomBg = 1;
nNegPerIm = 10;
overThres = 0.5;
negObjName = 'negBB';

invPading = 0;

verbose = 1;


%% initialization
close all;

addpath(genpath(vockitDir));
addpath(genpath(mhDir));


%% conversion
files = dir(sprintf('%s/*.png', srcImgDir));
pasAnns = cell(numel(files), 1);
parfor fInd=1:numel(files)
    imFN = files(fInd).name;
    imFFN = sprintf('%s/%s', srcImgDir, imFN);
    [~, imgID, ~] = fileparts(imFN);
    
    fprintf('- converting %d/%d...\n', fInd, numel(files));
    
    im = imread(imFFN);
    
    % convert to pascal format
    xmin = 1 + invPading;
    xmax = size(im, 2) - invPading;
    ymin = 1 + invPading;
    ymax = size(im, 1) - invPading;
    posBBs = [xmin; ymin; xmax; ymax];
    pasAnns{fInd} = convBB2Pascal( im, imFN, newDBName, dbName, posBBs, {ojCls} );
    
    % copy the image
    imwrite(im, [destImgDir '/' imgID '.jpg']);

    % save annotations
    curAnnFileNm = [imgID '.xml'];
    VOCwritexml(pasAnns{fInd}, [destAnnDir curAnnFileNm]);
    
end
pasAnns = cell2mat(pasAnns);

%% negs
files = dir(sprintf('%s/*.png', srcNegImgDir));
pasAnns_neg = cell(numel(files), 1);
parfor fInd=1:numel(files)
    imFN = files(fInd).name;
    imFFN = sprintf('%s/%s', srcNegImgDir, imFN);
    [~, imgID, ~] = fileparts(imFN);
    
    fprintf('- converting %d/%d...\n', fInd, numel(files));
    
    im = imread(imFFN);
    
    % convert to pascal format
    negBBs = genRandNegBB(size(im), [0 0 0 0]', nNegPerIm, overThres, minNegBBWH);
    negObjNames = cell(size(negBBs, 2), 1);
    [negObjNames{:}] = deal(negObjName);
    pasAnns_neg{fInd} = convBB2Pascal( im, imFN, newDBName, dbName, negBBs, negObjNames );
        
    
    % copy the image
    imwrite(im, [destImgDir '/' imgID '.jpg']);

    % save annotations
    curAnnFileNm = [imgID '.xml'];
    VOCwritexml(pasAnns_neg{fInd}, [destAnnDir curAnnFileNm]);
    
end
pasAnns_neg = cell2mat(pasAnns_neg);
pasAnns = [pasAnns; pasAnns_neg];



%% construct imagesets files
% train_inria

fidTr = fopen(sprintf('%strain_inria.txt', imgSetsDir), 'w');
for i=1:numel(pasAnns)
    [~, fn, ~] = fileparts(pasAnns(i).annotation.filename);
    
    fprintf(fidTr, '%s', fn);
    if i ~= numel(pasAnns)
        fprintf(fidTr, '\n');
    end    
end
fclose(fidTr);



%% show annotated results
if verbose == 1
    viewanno([imgSetsDir 'train_inria.txt'], destAnnDir, destImgDir);
end
