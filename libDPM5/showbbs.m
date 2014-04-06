% function showbbs(im, boxes, range, thres, cm, i_crop, i_showcb)
function showbbs(im, boxes, type, range, cm)
% Draw bounding boxes on top of an image.
%   showboxes(im, boxes, out)
%   If out is given, a pdf of the image is generated (requires export_fig).
%       type
%           0: root
%           1: root + part


nCM = size(cm, 1);
lineWidth = 3;
i_showcb = 1;
bbsize = 6;

image(im); 
axis image;
if isempty(range)
    switch type
        case 0
            range = [floor(min(boxes(:, end))), ceil(max(boxes(:, end)))];
        case 1
            range = [-1, 1];
    end
end


if isempty(boxes)
    return;
end

hold on;
nParts = (type==1)*((size(boxes, 2)-1)/bbsize) + (1-(type==1));
for bInd=size(boxes, 1):-1:1
    for pInd=1:nParts
        x1 = boxes(bInd, bbsize*(pInd-1)+1);
        y1 = boxes(bInd, bbsize*(pInd-1)+2);
        x2 = boxes(bInd, bbsize*(pInd-1)+3);
        y2 = boxes(bInd, bbsize*(pInd-1)+4);
        appScore = boxes(bInd, bbsize*(pInd-1)+5);
        defScore = boxes(bInd, bbsize*(pInd-1)+6);
        totalScore = boxes(bInd, end);
        
        % appeaerance
        switch type
            case 0
                score = totalScore;
            case 1
                score = appScore/abs(totalScore); % contribution of appearance score
        end
        score = min(range(2), max(range(1), score));

        colorInd = round((score-range(1))/(range(2) - range(1))*(nCM-1) + 1);
        c = cm(colorInd, :);
        
        line([x1 x1 x2 x2 x1]', [y1 y2 y2 y1 y1]', 'color', c, 'linewidth', lineWidth, 'linestyle', '-');
        
        % deformation
        switch type
            case 0
                
            case 1
                score = defScore/abs(totalScore); % contribution of deformation score
                score = min(range(2), max(range(1), score));
                colorInd = round((score-range(1))/(range(2) - range(1))*(nCM-1) + 1);
                c = cm(colorInd, :);
                
                plot(mean([x1, x2]), mean([y1, y2]), 'color', c, 'Marker', 'o', 'MarkerFaceColor', c, 'MarkerSize', 8);
        end        
    end
end
hold off;
if i_showcb == 1
    cb_h = colorbar;
    set(cb_h, 'YTick', [1 10:10:nCM]);
    set(cb_h, 'YTickLabel', [range(1):(range(2)-range(1))/6:range(2)]);
end

end