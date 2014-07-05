function [ o_mask ] = convBB2Mask( i_bb, i_imSz )
%CONVBB2MASK Summary of this function goes here
%   i_bb: [xmin ymin width height]
%   i_imSz: [W H]

x = [i_bb(1) i_bb(1) i_bb(1)+width i_bb(1)+width];
y = [i_bb(2) i_bb(2)+height i_bb(2)+height i_bb(2)];
o_mask = poly2mask(x, y, i_imSz(2), i_imSz(1));

end

