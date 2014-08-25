function [ o_pasRec ] = addObjPading( i_pasRec, i_objCls, i_padSize )
%ADDPADING Summary of this function goes here
%   Detailed explanation goes here

o_pasRec = i_pasRec;
if i_padSize == 0
    return;
end

parfor dbInd=1:numel(o_pasRec)
    % add paddings for pos examples
    posObjInd = find(strcmp(i_objCls, {o_pasRec(dbInd).objects(:).class}));
    for oInd=posObjInd(:)'
        o_pasRec(dbInd).objects(oInd).bbox(1:2) = o_pasRec(dbInd).objects(oInd).bbox(1:2) - i_padSize;
        o_pasRec(dbInd).objects(oInd).bbox(3:4) = o_pasRec(dbInd).objects(oInd).bbox(3:4) + i_padSize;
        
        o_pasRec(dbInd).objects(oInd).bndbox.xmin = o_pasRec(dbInd).objects(oInd).bbox(1);
        o_pasRec(dbInd).objects(oInd).bndbox.ymin = o_pasRec(dbInd).objects(oInd).bbox(2);
        o_pasRec(dbInd).objects(oInd).bndbox.xmax = o_pasRec(dbInd).objects(oInd).bbox(3);
        o_pasRec(dbInd).objects(oInd).bndbox.ymax = o_pasRec(dbInd).objects(oInd).bbox(4);
    end
end

end

