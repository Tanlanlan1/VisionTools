function [ C ] = myconv2( A, B, shape, maxWH )
%MYCONV2 Summary of this function goes here
%   Detailed explanation goes here

%% init
if nargn == 3
    maxWH = [640; 480];
end
%%FIXME: assume A is larger than B

%% change scale if it's too big
maxPix = maxWH(1)*maxWH(2);
curPix = size(A, 1)*size(A, 2);
if curPix > maxPix
    S = curPix/maxPix;
    A_res = imresize(A, 1/S, 'bicubic');
    B_res = imresize(B, 1/S, 'bicubic');
else
    A_res = A;
    B_res = B;
end

%% run conv2
C_res = conv2(A_res, B_res, shape);

%% rescale
if curPix > maxPix
    C = imresize(C_res, S, 'bicubic');
    if strcmp(shape, 'same') %%FIXME: not good choice
        C = imresize(C, [size(A, 1) size(A, 2)], 'bicubic');
    elseif strcmp(shape, 'valid')
        C = imresize(C, [size(A, 1)-(size(B, 1)-1), size(A, 2)-(size(B, 2)-1)]);
    else
        warning('Not yet implemented');
    end
else
    C = C_res;
end
end

