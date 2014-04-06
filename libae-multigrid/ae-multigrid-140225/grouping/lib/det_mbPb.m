function [bg1, bg2, bg3] = det_mbPb(im, cacheFN)
% compute image gradients. Implementation by Michael Maire.

if nargin<2
    cacheFN = [];
end

if exist(cacheFN, 'file')
    load(cacheFN);
%     bg1 = double(bg1);
%     bg2 = double(bg2);
%     bg3 = double(bg3);
%     cga1 = double(cga1);
%     cga2 = double(cga2);
%     cga3 = double(cga3);
%     cgb1 = double(cgb1);
%     cgb2 = double(cgb2);
%     cgb3 = double(cgb3);
%     tg1 = double(tg1);
%     tg2 = double(tg2);
%     tg3 = double(tg3);
%     textons = double(textons);
    return;
end

% compute pb parts
[bg_r3, bg_r5, bg_r10] = mex_pb_parts_final_selected_b(im(:,:,1),im(:,:,2),im(:,:,3));
    
[sx sy sz] = size(im);
temp = zeros([sx sy 8]);

for r = [3 5 10]
    for ori = 1:8
        eval(['temp(:,:,ori) = bg_r' num2str(r) '{' num2str(ori) '};']);
    end
    eval(['bg_r' num2str(r) ' = temp;']);
end
bg1 = bg_r3; bg2 = bg_r5;  bg3 = bg_r10; 


% caching
% bg1 = single(bg1);
% bg2 = single(bg2);
% bg3 = single(bg3);
% cga1 = single(cga1);
% cga2 = single(cga2);
% cga3 = single(cga3);
% cgb1 = single(cgb1);
% cgb2 = single(cgb2);
% cgb3 = single(cgb3);
% tg1 = single(tg1);
% tg2 = single(tg2);
% tg3 = single(tg3);
% textons = single(textons);
if ~isempty(cacheFN)
    save(cacheFN, 'bg1', 'bg2', 'bg3');
end

end