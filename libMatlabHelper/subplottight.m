function subplottight(n,m,i)
    [c,r] = ind2sub([m n], i);
    padSize = 5*1e-2;
    subplot('Position', [(c-1)/m + padSize, 1-(r)/n + padSize, 1/m - padSize*2, 1/n - padSize*2]);
end