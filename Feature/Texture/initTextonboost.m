function [ o_tbParams_parts ] = initTextonboost( i_tbParams_layoutFilterWH, i_tbParams_nParts, i_debugOpt )
%INITTEXTONBOOST Summary of this function goes here
%   Detailed explanation goes here

o_tbParams_parts = [];
tbWH = i_tbParams_layoutFilterWH;
for pInd=1:i_tbParams_nParts
    xs = [0; 0];
    while xs(1) == xs(2)
        xs = randi(tbWH(1), [2, 1]);
    end
    
    ys = [0; 0];
    while ys(1) == ys(2)
        ys = randi(tbWH(2), [2, 1]);
    end
    
    o_tbParams_parts = [o_tbParams_parts [min(xs); max(xs); min(ys); max(ys)]];
end

if i_debugOpt >= 1
    figure(34125); clf;
    rectangle('Position', [1 1 tbWH(:)']); hold on; % layoutFilter
    for pInd=1:i_tbParams_nParts
        subWin = o_tbParams_parts(:, pInd)';
        rectangle('Position', [subWin(1) subWin(3) subWin([2,4]) - subWin([1 3])]); hold on; % subwindows
    end
end

end

