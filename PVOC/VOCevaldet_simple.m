
% function [rec,prec,ap] = VOCevaldet_simple(VOCopts,id,cls,draw)
function [recall_gt, prec, ap, o_pasDB_tp, o_pasDB_fp] = VOCevaldet_simple(pasDB_gt, pasDB_det, cls, minoverlap, draw)

assert(numel(pasDB_gt) == numel(pasDB_det));

%% extract detections
dbObjInd = [];
gts = [];
BBs = [];
scores = [];
npos = 0;
for dbInd=1:numel(pasDB_det)
    
    % handle gt
    rec_gt = pasDB_gt(dbInd);
    gt = [];
    clsinds = find(strcmp(cls, {rec_gt.objects(:).class}));
    gt.BB = cat(1, rec_gt.objects(clsinds).bbox)';
    gt.diff = [rec_gt.objects(clsinds).difficult];
    gt.det = false(length(clsinds),1);
    npos = npos+sum(~gt.diff);

    gts = [gts; gt];
    
    % handle det
    rec_det = pasDB_det(dbInd);
    if ~isempty(rec_det.objects) %%FIXME: correct?
        clsinds = find(strcmp(cls, {rec_det.objects(:).class}));
        BB = cat(1, rec_det.objects(clsinds).bbox)';
        score = cat(1, rec_det.objects(clsinds).score)';
        BBs = [BBs BB];
        scores = [scores score];

        % remember indice
        dbObjInd = [dbObjInd [ones(1, numel(clsinds))*dbInd; find(clsinds)]];
    end
end


%% calc tp/fp    
[~, si] = sort(scores, 'descend');
BB = BBs(:, si);
score = scores(si);

% assign detections to ground truth objects
nd = length(score);
tp = zeros(nd,1);
fp = zeros(nd,1);

for d=1:nd
    % assign detection to ground truth object if any
    bb = BB(:,d);
    gt = gts(dbObjInd(1, si(d)));

    ovmax=-inf;
    for j=1:size(gt.BB,2)
        bbgt = gt.BB(:,j);
        bi = [max(bb(1),bbgt(1)) ; max(bb(2),bbgt(2)) ; min(bb(3),bbgt(3)) ; min(bb(4),bbgt(4))];
        iw=bi(3)-bi(1)+1;
        ih=bi(4)-bi(2)+1;
        if iw>0 && ih>0                
            % compute overlap as area of intersection / area of union
            ua=(bb(3)-bb(1)+1)*(bb(4)-bb(2)+1)+...
               (bbgt(3)-bbgt(1)+1)*(bbgt(4)-bbgt(2)+1)-...
               iw*ih;
            ov=iw*ih/ua;
            if ov>ovmax
                ovmax=ov;
                jmax=j;
            end
        end
    end
    % assign detection as true positive/don't care/false positive
    if ovmax>=minoverlap
        if ~gt.diff(jmax)
            if ~gt.det(jmax)
                tp(d)=1;            % true positive
                gt.det(jmax)=true;
            else
                fp(d)=1;            % false positive (multiple detection)
            end
        end
    else
        fp(d)=1;                    % false positive
    end

    gts(dbObjInd(1, si(d))) = gt;
end

    
%% convert to pascal DB
o_pasDB_tp = pasDB_det;
o_pasDB_fp = pasDB_det;
bbtp = si(boolean(tp));
bbfp = si(boolean(fp));
for gtInd=1:numel(pasDB_gt)
    curDBInd = find(dbObjInd(1, :) == gtInd);
    
    tpObjInd = dbObjInd(2, intersect(curDBInd, bbtp));
    o_pasDB_tp(gtInd).objects = o_pasDB_tp(gtInd).objects(tpObjInd);
    
    fpObjInd = dbObjInd(2, intersect(curDBInd, bbfp));
    o_pasDB_fp(gtInd).objects = o_pasDB_fp(gtInd).objects(fpObjInd);
end

    
% end
% tp = cell2mat(tpCell);
% fp = cell2mat(fpCell);
        
% compute precision/recall
fp = cumsum(fp);
tp = cumsum(tp);
recall_gt = tp/npos;
prec=tp./(fp+tp);

ap=VOCap(recall_gt,prec);

if draw
    % plot precision/recall
    plot(recall_gt,prec,'-');
    grid;
    xlabel 'recall'
    ylabel 'precision'
    title(sprintf('class: %s, AP = %.3f',cls, ap));
    xlim([0 1]);
    ylim([0 1]);
end

end