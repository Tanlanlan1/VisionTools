function [ o_pasDB ] = convPasObjCls( i_pasDB, i_fromCls, i_toCls )
%CONVPASOBJCLS Summary of this function goes here
%   Detailed explanation goes here

o_pasDB = i_pasDB;
for dbInd=1:numel(i_pasDB)
    objs = i_pasDB(dbInd).objects;
    for oInd=1:numel(objs)
        if strcmp(objs(oInd).class, i_fromCls)
            objs(oInd).class = i_toCls;
        end
    end
    o_pasDB(dbInd).objects = objs;
end
end

