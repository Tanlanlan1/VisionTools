%% user settings
vockitDir = '../libPascal';
mhDir = '../libMatlabHelper';

dbName = 'Caltech101';
newDBName = ['VOC_' dbName];
srcDBDir = ['/data/v50/sangdonp/objectDetection/' dbName];
srcImgDir = [srcDBDir '/101_ObjectCategories'];
srcAnnDir = [srcDBDir '/Annotations'];
% srcAnnoDir_mot = '/data/v50/sangdonp/objectDetection/caltech_motorbikes_side';

destDBDir = ['/data/v50/sangdonp/objectDetection/' newDBName];

objClss = {'car_side', 'Motorbikes'};
% objClss = {'car_side'};

minNegBBWH = [2, 2];

padding = false;
trainingRatio = 0.25;
validateRatio = 0.25;
testRatio = 0.5;
preferImgArea = 200*100;
bgImgArea = 400*400;

genRandomBg = 1;
nNegPerIm = 10;
overThres = 0.5;
negObjName = 'negBB';

synthWithBg = true;
srcBgImgDir = [srcImgDir '/BACKGROUND_Google'];

verbose = 1;


%% initialization
close all;

addpath(genpath(vockitDir));
addpath(genpath(mhDir));

if ~exist(destDBDir, 'dir')
    mkdir(destDBDir);
end

imgDir_pas = [destDBDir '/JPEGImages/'];
annDir_pas = [destDBDir '/Annotations/'];
imgSetsDir = [destDBDir '/ImageSets/Main/'];

if exist(imgDir_pas, 'dir')
    rmdir(imgDir_pas, 's');
end
if exist(annDir_pas, 'dir')
    rmdir(annDir_pas, 's');
end
if exist(imgSetsDir, 'dir')
    rmdir(imgSetsDir, 's');
end

mkdir(imgDir_pas);
mkdir(annDir_pas);
mkdir(imgSetsDir);

%% bg
bgFiles = dir(sprintf('%s/*.jpg', srcBgImgDir));

%% conversion
pasAnns = [];
for objClsInd=1:numel(objClss)
    curObjCls = objClss{objClsInd};

    % load source db
    files = dir(sprintf('%s/%s/*.jpg', srcImgDir, curObjCls));
    curPasAnns = cell(numel(files), 1);
    parfor fInd=1:numel(files)
        imFN = files(fInd).name;
        imFN_new = [curObjCls '_' imFN];
        imFFN = sprintf('%s/%s/%s', srcImgDir, curObjCls, imFN);
        imgID = imFN(7:strfind(imFN, '.')-1);

        fprintf('- converting (%d/%d, %d/%d)...\n', objClsInd, numel(objClss), fInd, numel(files));

        % load the annotations
        annFN = sprintf('%s/%s/annotation_%s.mat', srcAnnDir, curObjCls, imgID);
        annSt = load(annFN); % box_coord: [ymin ymax xmin xmax] 
        calBBs = annSt.box_coord;
        calConts = annSt.obj_contour;
        
        % convert images
        im = gray2rgb(imread(imFFN));
        
        % scale images
        if preferImgArea > 0
            curScale = sqrt(preferImgArea/(size(im, 1)*size(im, 2)));
            im = imresize(im, curScale);
        else
            curScale = 1;
        end
        calBBs = calBBs*curScale;
        calConts = calConts*curScale;
        
        % synthesis background
        if synthWithBg
            
            rndBgFN = bgFiles(randi(numel(bgFiles), 1));
            bgImFN = sprintf('%s/%s', srcBgImgDir, rndBgFN.name);
            bgImg = gray2rgb(imread(bgImFN));
            bgImg_resc = imresize(bgImg, sqrt(bgImgArea/prod([size(bgImg, 1), size(bgImg, 2)])));
            
%             if size(bgImg_resc, 1) > 800 || size(bgImg_resc, 2) > 800
%                 keyboard;
%             end
            [im, poly_mod] = imfbsynthesis(im, bgImg_resc, [calConts(1, :)+calBBs(3); calConts(2, :)+calBBs(1)]);
            calBBs = [min(poly_mod(2, :)), max(poly_mod(2, :)) min(poly_mod(1, :)) max(poly_mod(1, :))];
        end
        
        % create padded images
        if padding == 1
            xPading = size(im, 2);
            yPading = size(im, 1);
            im = padarray(im, round([yPading/2 xPading/2]), 'replicate');
            calBBs(3) = calBBs(3) + round(xPading/2);
            calBBs(4) = calBBs(4) + round(xPading/2);
            calBBs(1) = calBBs(1) + round(yPading/2);
            calBBs(2) = calBBs(2) + round(yPading/2);
        end
        
        % convert to pascal format
        xmin = calBBs(3);
        xmax = calBBs(4);
        ymin = calBBs(1);
        ymax = calBBs(2);
        posBBs = [xmin; ymin; xmax; ymax];
        posPasDB = convBB2Pascal( im, imFN_new, newDBName, dbName, posBBs, {curObjCls} );

        % convert negative examples
        if genRandomBg == 1
            negBBs = genRandNegBB(size(im), posBBs, nNegPerIm, overThres, minNegBBWH);
            negObjNames = cell(size(negBBs, 2), 1);
            [negObjNames{:}] = deal(negObjName);
            negPasDB = convBB2Pascal( im, imFN_new, newDBName, dbName, negBBs, negObjNames );

            % merge positives and negatives
            pasDB = [];
            pasDB.annotation = mergePascalDB(posPasDB.annotation, negPasDB.annotation);
        else
            pasDB = posPasDB;
        end
        
        % copy images
        imwrite(im, [imgDir_pas '/' imFN_new]);
    
        % save annotations
        curAnnFileNm = strrep(imFN_new, 'jpg', 'xml');
        VOCwritexml(pasDB, [annDir_pas curAnnFileNm]);

        % save
        curPasAnns{fInd} = pasDB;
    end
    pasAnns = [pasAnns; curPasAnns];
end
nAnnDB = numel(pasAnns);
pasAnns = cell2mat(pasAnns);

%% separate a training/validate/test set
nTraining = round(nAnnDB*trainingRatio);
nTest = round(nAnnDB*testRatio);
nValid = round(nAnnDB*validateRatio);

perInd = randperm(nAnnDB);
trainingInd = perInd(1:nTraining);
validInd = perInd(nTraining+1:nTraining+nValid);
testInd = perInd(nTraining+nValid+1:end);

%% construct imagesets files
% trainval
tvInd = 0;
fidTV = fopen(sprintf('%strainval.txt', imgSetsDir), 'w');

% train
fidTr = fopen(sprintf('%strain.txt', imgSetsDir), 'w');
for i=1:numel(trainingInd)
    trInd = trainingInd(i);
    [~, fn, ~] = fileparts(pasAnns(trInd).annotation.filename);
    
    fprintf(fidTr, '%s', fn);
    if i ~= numel(trainingInd)
        fprintf(fidTr, '\n');
    end
    
    % trainval
    fprintf(fidTV, '%s', fn);
    tvInd = tvInd + 1;
    if tvInd ~= numel(trainingInd)+numel(validInd)
        fprintf(fidTV, '\n');
    end
    
end
fclose(fidTr);

% validate
fidVal = fopen(sprintf('%sval.txt', imgSetsDir), 'w');
for i=1:numel(validInd)
    valInd = validInd(i);
    [~, fn, ~] = fileparts(pasAnns(valInd).annotation.filename);
    
    fprintf(fidVal, '%s', fn);
    if i ~= numel(validInd)
        fprintf(fidVal, '\n');
    end
    
    % trainval
    fprintf(fidTV, '%s', fn);
    tvInd = tvInd + 1;
    if tvInd ~= numel(trainingInd)+numel(validInd)
        fprintf(fidTV, '\n');
    end
end
fclose(fidVal);

% trainval
% copyfile(sprintf('%strain.txt', imgSetsDir), sprintf('%strainval.txt', imgSetsDir));
fclose(fidTV);

% test
fidTest = fopen(sprintf('%stest.txt', imgSetsDir), 'w');
for i=1:numel(testInd)
    teInd = testInd(i);
    [~, fn, ~] = fileparts(pasAnns(teInd).annotation.filename);
    
    fprintf(fidTest, '%s', fn);
    if i ~= numel(testInd)
        fprintf(fidTest, '\n');
    end
end
fclose(fidTest);


%% show annotated results
if verbose == 1
    viewanno([imgSetsDir '/train.txt'], annDir_pas, imgDir_pas);
    viewanno([imgSetsDir '/val.txt'], annDir_pas, imgDir_pas);
    viewanno([imgSetsDir '/test.txt'], annDir_pas, imgDir_pas);
end
