function [ o_label ] = GetSuperpixel( i_img, i_method, i_params )
% 
%   Matlab wrapper of superpixel methods
%   
% ----------
%   Input: 
% 
%       i_img:          an image
%       i_method:       'NCut', 'Turbopixel', 'SLIC'
%       i_params:       parameter structure for each methods
%           'NCut'              i_params.N: the number of segments
%                               i_params.verbosity: the level of verbosity [0 (slient), 1]
% 
%           'Turbopixel'        i_params.N: the number of segments
%                               i_params.verbosity: the level of verbosity [0 (slient), 1]
% 
%           'SLIC'              i_params.N: the number of segments
%                               i_params.verbosity: the level of verbosity [0 (slient), 1]
% 
% ----------
%   Output:
% 
%       o_label:        segment labels
% 
% ----------
%   DEPENDENCY:
%   
%
% ----------
% Written by Sangdon Park (sangdonp@cis.upenn.edu), 2014.
% All rights reserved.
%

%% init
assert(isfield(i_params, 'N'));
if ~isfield(i_params, 'verbosity')
    i_params.verbosity = 0;
end



%% run a superpixel algorithm
switch i_method
    case 'NCut'
        [Inr, Inc, nd] = size(i_img);
        if (nd>1),
            I = im2double(rgb2gray(i_img));
        else
            I = im2double(i_img);
        end
        I(I>1) = 1;
        I(I<0) = 0;
        I = imresize(I, [160, 160], 'bicubic');
        addpath('../Ncut_9');
        [SegLabel,NcutDiscrete,NcutEigenvectors,NcutEigenvalues,W,imageEdges]= NcutImage(I,i_params.N);
        if i_params.verbosity >= 1
            figure(30000);
            bw = edge(SegLabel,0.01);
            J1 = showmask(I,imdilate(bw,ones(2,2))); imagesc(J1);axis off
        end
        rmpath('../Ncut_9');
        
        o_label = imresize(SegLabel, [Inr, Inc], 'nearest');
        
    case 'Turbopixel'
        addpath('../TurboPixels');
        [phi,boundary,disp_img, sup_image] = superpixels(im2double(i_img), i_params.N);
        rmpath('../TurboPixels');
        
        o_label = sup_image;
        
        if i_params.verbosity >= 1
            figure(30000);
            imagesc(disp_img);
        end
        
    case 'SLIC'
end

end

