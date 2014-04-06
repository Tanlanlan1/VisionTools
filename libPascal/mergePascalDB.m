function [ o_merPas ] = mergePascalDB( i_toPas, i_fromPas )
%MERGEPASCAL Summary of this function goes here
%   Detailed explanation goes here
if isfield(i_toPas, 'annotation')
    i_toPasRec = i_toPas.annotation;
else
    i_toPasRec = i_toPas;
end
if isfield(i_fromPas, 'annotation')
    i_fromPasRec = i_fromPas.annotation;
else
    i_fromPasRec = i_fromPas;
end
assert(numel(i_toPasRec) == numel(i_fromPasRec));

objTmp = [];
for dbInd=1:numel(i_toPasRec)
    if numel(i_toPasRec(dbInd).objects) > 0
        objTmp = i_toPasRec(dbInd).objects(1);
        break;
    end
    if numel(i_fromPasRec(dbInd).objects) > 0
        objTmp = i_fromPasRec(dbInd).objects(1);
        break;
    end
end
assert(~isempty(objTmp));

o_merPasRec = i_toPasRec;
for dbInd=1:numel(o_merPasRec)
    fromObjs = i_fromPasRec(dbInd).objects(:);
    obj = objTmp;
    for oInd=1:numel(fromObjs)

        curObj = fromObjs(oInd);
        assert(isfield(curObj, 'bbox'));
        
        obj.class = curObj.class;
        obj.bbox = curObj.bbox;
        obj.bndbox.xmin = curObj.bbox(1);
        obj.bndbox.ymin = curObj.bbox(2);
        obj.bndbox.xmax = curObj.bbox(3);
        obj.bndbox.ymax = curObj.bbox(4);
        
        o_merPasRec(dbInd).objects = [o_merPasRec(dbInd).objects(:); obj];
    end
end

o_merPas = [];
o_merPas.annotation = o_merPasRec;
end

