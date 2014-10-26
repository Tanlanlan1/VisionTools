function [ o_label, o_label_sub ] = GetSuperpixel( i_img, i_method, i_params )
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
%           'TurboPixel'        i_params.N: the number of segments
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
if nargin == 2
    i_params = [];
end
if ~isfield(i_params, 'verbosity')
    i_params.verbosity = 0;
end



%% run a superpixel algorithm
switch i_method
    case 'NCut'
        assert(isfield(i_params, 'N'));
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
        
    case 'TurboPixel'
        assert(isfield(i_params, 'N'));
        addpath(genpath('../TurboPixels'));
        [phi,boundary,disp_img, sup_image] = superpixels(im2double(i_img), i_params.N);
        rmpath(genpath('../TurboPixels'));
        
        o_label = sup_image;
        
        if i_params.verbosity >= 1
            figure(30000);
            imagesc(disp_img);
        end
        
    case 'SLIC'
        %% set default values
        if ~isfield(i_params, 'regionSize')
            i_params.regionSize = 20;
        end
        if ~isfield(i_params, 'regularizer')
            i_params.regularizer = 0.1;
        end
        
        %% add path
        thisFilePath = fileparts(mfilename('fullpath'));
        vlfeatmexpath = [thisFilePath '/../vlfeat/toolbox/'];
        vlfeatmexapthall = genpath(vlfeatmexpath);
        addpath(vlfeatmexapthall);
        if ~strfind(getenv('LD_LIBRARY_PATH'), vlfeatmexapthall)
            setenv('LD_LIBRARY_PATH', [vlfeatmexapthall ':' getenv('LD_LIBRARY_PATH')]);
        end
        %% run
        imlab = vl_xyz2lab(vl_rgb2xyz(min(max(0, im2double(i_img)), 1)));
        o_label = vl_slic(single(imlab), i_params.regionSize, i_params.regularizer);
        
        uLabels = unique(o_label(:)');
        o_label_sub = cell(1, numel(uLabels));
        for lInd=1:numel(uLabels)
            [rs, cs] = find(o_label == uLabels(lInd));
            o_label_sub{lInd} = [cs(:)'; rs(:)'];
        end
        
        if i_params.verbosity >= 1
            figure(30000);
            imagesc(im2double(i_img));
            hold on;
            h = imagesc(imdilate(edge(double(o_label)/double(max(o_label(:))), 'sobel', 0), ones(2)));
            axis image;
            set(h, 'AlphaData', 0.7);
            hold off;
            
        end
        
        %% rm path
        rmpath(vlfeatmexapthall);
    otherwise
        warning('* Incorrect method: %s', i_method);
end

end

