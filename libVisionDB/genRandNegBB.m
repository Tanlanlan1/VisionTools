function [ o_negBBs ] = genRandNegBB( i_imSize, i_posBBs, i_nNeg, i_overThres, i_minNegBBWH )
%GENRANDNEGBB Summary of this function goes here
%   Detailed explanation goes here
width = i_imSize(2);
height = i_imSize(1);

curNegInd = 1;
o_negBBs = zeros(4, i_nNeg);
while 1
    if curNegInd > i_nNeg
        break;
    end
    xs = rand(2,1)*width;
    ys = rand(2,1)*height;
    if abs(xs(1) - xs(2)) < i_minNegBBWH(1)
        continue;
    end
    if abs(ys(1) - ys(2)) < i_minNegBBWH(2)
        continue;
    end
    
    curNegBB = [min(xs); min(ys); max(xs); max(ys)];
    isValid = true;
    for pInd=1:size(i_posBBs, 2)
        curPosBB = i_posBBs(:, pInd);
        intarea = rectint(...
            [curPosBB(1), curPosBB(2), curPosBB(3)-curPosBB(1), curPosBB(4)-curPosBB(2)], ...
            [curNegBB(1), curNegBB(2), curNegBB(3)-curNegBB(1), curNegBB(4)-curNegBB(2)]);
        
        IOU = intarea/(prod([curPosBB(3)-curPosBB(1), curPosBB(4)-curPosBB(2)]) + prod([curNegBB(3)-curNegBB(1), curNegBB(4)-curNegBB(2)]) - intarea);
        if IOU > i_overThres
            isValid = false;
            break;
        end
    end
    
    if isValid
        o_negBBs(:, curNegInd) = curNegBB;
        curNegInd = curNegInd + 1;
    end
    
end

end

