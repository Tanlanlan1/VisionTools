function [o_pasDB] = loadPascalDB(i_params, i_imgSet )
% nMaxNegPerImg = i_params.training.nMaxNegPerImg;
dbIDs = textscan(fopen(i_imgSet), '%s');
dbIDs = dbIDs{1};
nTrDB = numel(dbIDs);

pasDB = cell(nTrDB, 1);
parfor dbInd=1:nTrDB
    curAnnFN = sprintf('%s/%s.xml', i_params.db.annDir, dbIDs{dbInd});
    pasDB{dbInd} = PASreadrecord(curAnnFN);
end
o_pasDB = cell2mat(pasDB);

end
