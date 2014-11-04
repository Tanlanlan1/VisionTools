function [out] = myrectint(A, B)
% modified matlab code for efficiency
assert(size(A, 1) == 1 && size(B, 1) == 1);

leftA = A(:,1);
bottomA = A(:,2);
rightA = leftA + A(:,3);
topA = bottomA + A(:,4);

leftB = B(:,1)';
bottomB = B(:,2)';
rightB = leftB + B(:,3)';
topB = bottomB + B(:,4)';

out = (max(0, min(rightA, rightB) - max(leftA, leftB))) .* ...
    (max(0, min(topA, topB) - max(bottomA, bottomB)));
end