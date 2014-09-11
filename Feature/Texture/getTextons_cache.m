function [ o_textons ] = getTextons_cache( i_trainImgs, i_sampleStep, i_K, i_fb, i_cacheFN )
%GETTEXTON Summary of this function goes here
%   Detailed explanation goes here

if exist(i_cacheFN, 'file')
    load(i_cacheFN);
else
    
    %% obtain textons
    o_textons = getTextons( i_trainImgs, i_sampleStep, i_K, i_fb );
    
    %% save
    save('textons.mat', 'o_textons');
end
%% visualize
fb_c = {};
for i=1:size(i_fb, 3)
    fb_c{i} = i_fb(:, :, i);
end
[tim,tperm] = visTextons(o_textons, fb_c);
figure(20001); clf;
imgarray = cell2mat(reshape({tim{tperm}}, [1 1 1 i_K]));
montage(cell2mat(reshape({tim{tperm}}, [1 1 1 i_K])), 'DisplayRange', [min(imgarray(:)), max(imgarray(:))])
axis image; colorbar;


end

