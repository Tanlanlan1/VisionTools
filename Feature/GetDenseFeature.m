function [ o_feat ] = GetDenseFeature( i_img, i_cues, i_params )
% 
%   Matlab wrapper of dense feature extract methods
%   
% ----------
%   Input: 
% 
%       i_img:          a RGB image
%       i_cues:         a string array where each elements stands for cues, e.g. Lab, texture
%           'color_RGB'         use RGB colors
%           'color_Lab'         use Lab colors
%           'texture_LM'        use the Leung-Malik filter bank
% 
%       i_params:       parameter structure for each cues
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
%       o_feat:        dense features
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
addpath('./Texture');
if nargin < 3
    i_params = struct('verbosity', 0);
end
if ~isfield(i_params, 'verbosity')
    i_params.verbosity = 0;
end

%% extract features
img = im2double(i_img);
o_feat = cell(1, 1, numel(i_cues));
for cInd=1:numel(i_cues)
    
    switch i_cues{cInd}
        case 'color_RGB'
            assert(size(img, 3) == 3);
            o_feat{cInd} = img;
            
        case 'color_Lab'
            assert(size(img, 3) == 3);    
            o_feat{cInd} = applycform(img, makecform('srgb2lab'));
            
        case 'texture_LM'
            img = rgb2gray(img);
            Fs = makeLMfilters;
            img_pad = padarray(img, [(size(Fs, 1)-1)/2 (size(Fs, 2)-1)/2], 'symmetric', 'both');
            responses = zeros(size(img, 1), size(img, 2), size(Fs, 3));
            for fInd=1:size(Fs, 3)
                responses(:, :, fInd) = conv2(img_pad, Fs(:, :, fInd), 'valid'); % symetric filters, so don't need to flip
            end
            o_feat{cInd} = responses;
            
        case 'texture_MR4'
            img = rgb2gray(img);
            
            
        otherwise
            warning('Wrong cue name: %s', i_cues{cInd});
    end
    
end

%% return
o_feat = cell2mat(o_feat);

%% show
if i_params.verbosity >= 1
    figure(50000);
    for fInd=1:size(o_feat, 3)
        imagesc(o_feat(:, :, fInd));
        axis image;
        colorbar;
        pause(0.1);
    end
end

end

