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


%% run a superpixel algorithm
switch i_method
    case 'NCut'
        
    case 'Turbopixel'
        
        [phi,boundary,disp_img, sup_image] = superpixels(im2double(i_img), 500);
        o_label = sup_image;
        
        if i_params.verbosity >= 1
            imagesc(disp_img);
        end
        
    case 'SLIC'
end

end

