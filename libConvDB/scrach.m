imgDir_src = '/data/v50/sangdonp/FacadeRecognition/data/windows_before/Images';
annDir_src = '/data/v50/sangdonp/FacadeRecognition/data/windows_before/Annotations';
imgDir_des = '/data/v50/sangdonp/FacadeRecognition/data/windows/Images';
annDir_des = '/data/v50/sangdonp/FacadeRecognition/data/windows/Annotations';

mkdir(imgDir_des);
mkdir(annDir_des);

files = dir(imgDir_src);
imgID = 1;
for fInd=1:numel(files)
    if files(fInd).isdir 
        continue;
    end
    fileNewID = sprintf('%.4d', imgID);
    
    [~, fileID, ~] = fileparts(files(fInd).name);
    img = imread([imgDir_src '/' files(fInd).name]);
    imwrite(img, [imgDir_des '/' fileNewID '.png']);
    
    
    if exist([annDir_src '/' fileID '.mat'], 'file')
        copyfile([annDir_src '/' fileID '.mat'], [annDir_des '/' fileNewID '.mat']);
    end
    imgID = imgID + 1;
end