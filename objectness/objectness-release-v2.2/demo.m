% imgExample = imread('002053.jpg');
im = imread('test1.png');
tic;
boxes = runObjectness(im,1000);
toc;
figure,imshow(im),drawBoxes(boxes);

% for i=1:size(boxes,1)
%     bbox = boxes(i, :);
%     figure(2);
%     imshow(im);
%     hold on; 
%     rectangle('Position', [bbox(2) bbox(1) bbox(4)-bbox(2) bbox(3)-bbox(1)], 'EdgeColor', 'r', 'lineWidth', 4);
%     hold off;
%     saveas(2, sprintf('%s/%d.png',  'tmp1', i)); 
% end
