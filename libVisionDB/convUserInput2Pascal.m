%% user settings
% vockitDir = '../libPascal';
vockitDir = '~/UPenn/Dropbox/Research/tools/libPascal';
dbName = 'RCTA';

% srcDBDir = 
srcImgDir = '/data/v50/sangdonp/FacadeRecognition/data/RCTA_VOC/';


destDBDir = ['/data/v50/sangdonp/FacadeRecognition/data/' dbName '_VOC'];
posObjName = 'door';


minNegBBWH = [2, 2];
uniformNegSize = 1;

% padding = 1;
trainingRatio = 1;
validateRatio = 0;
testRatio = 0;
preferImgArea = 0;

genRandomBg = 1;
nNegPerIm = 100;
overThres = 0.8;
negObjName = 'randBB';
imgFileExt = 'jpg';
annFileExt = 'mat';

show = 1;

%% initialization
addpath(genpath(vockitDir));

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


%% conversion


% load source db
files = dir(sprintf('%s/*.%s', srcImgDir, imgFileExt));
pasAnns = cell(numel(files), 1);
for fInd=1:numel(files)
    
    imFN = files(fInd).name;
    imFFN = sprintf('%s/%s', srcImgDir, imFN);
%     imFN_new = [posObjName '_' imFN];
    [~, imgID, ~] = fileparts(imFN);
    imFN_new = sprintf('%s.jpg', imgID);
    
    
    fprintf('- converting img %s (%d/%d)...\n', imFN, fInd, numel(files));
    
    %% convert positive examples
    im = imread(imFFN);
    if preferImgArea > 0
        curScale = sqrt(preferImgArea/(size(im, 1)*size(im, 2)));
        im = imresize(im, curScale);
    else
        curScale = 1;
    end
    
    % annotate
    figure(10001);
    imshow(im);
    axis on; axis image;

    polys = [];
    while 1
        [x, y] = ginput(4);
        if isempty(x)
            break;
        end

        hold on;
        rectangle('Position', [min(x) min(y) max(x)-min(x) max(y)-min(y)], 'EdgeColor', 'g')
        hold off;

        polys = [polys; {ceil([x'; y'])}];
    end
    
    
    posBBs = [];
    for pInd=1:numel(polys)
        curPoly = ceil(polys{pInd});
        xmin = min(curPoly(1, :))*curScale;
        ymin = min(curPoly(2, :))*curScale;
        xmax = max(curPoly(1, :))*curScale;
        ymax = max(curPoly(2, :))*curScale;
        
        posBBs = [xmin'; ymin'; xmax'; ymax'];
        

    end
    
    posObjNames = cell(size(posBBs, 2), 1);
    [posObjNames(:)] = deal({posObjName});
    posPasDB = convBB2Pascal( im, imFN_new, dbName, dbName, posBBs, posObjNames );
    
    % convert negative examples
    if genRandomBg == 1
        
        if uniformNegSize == 1
            assert(numel(posPasDB(1).annotation.objects) == 1);
            negWH = [posBBs(3)-posBBs(1); posBBs(4)-posBBs(2)];
        else
            negWH = [];
        end
            
        negBBs = genRandNegBB(size(im), posBBs, nNegPerIm, overThres, minNegBBWH, negWH);
        negObjNames = cell(size(negBBs, 2), 1);
        [negObjNames{:}] = deal(negObjName);
        negPasDB = convBB2Pascal( im, imFN_new, dbName, dbName, negBBs, negObjNames );

        % merge positives and negatives
        pasDB = [];
        pasDB = mergePascalDB(posPasDB, negPasDB);
    else
        pasDB = posPasDB;
    end
    
    % copy images
    imwrite(im, [imgDir_pas '/' imFN_new]);

    % save annotations
    curAnnFileNm = sprintf('%s.%s', imgID, 'xml');
    VOCwritexml(pasDB, [annDir_pas curAnnFileNm]);
    
    % save
    pasAnns{fInd} = pasDB;
    
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
if show == 1
    viewanno([imgSetsDir '/train.txt'], annDir_pas, imgDir_pas);
    viewanno([imgSetsDir '/val.txt'], annDir_pas, imgDir_pas);
    viewanno([imgSetsDir '/test.txt'], annDir_pas, imgDir_pas);
end
