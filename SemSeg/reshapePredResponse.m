function [o_resp] = reshapePredResponse(i_nImgs, i_ixy, i_imgWHs, i_resp, i_supLbl)

respSt = struct('resp', zeros(0, 0, 0, 0));
o_resp = repmat(respSt, [i_nImgs, 1]);
coder.varsize('o_resp(:).resp');


nCls = size(i_resp, 2);
nClf = size(i_resp, 3);
for iInd=1:i_nImgs
    
    resp_resh = zeros(i_imgWHs(2, iInd), i_imgWHs(1, iInd), nCls, nClf);
    for cfInd=1:nClf
        for cInd=1:nCls
            curResp = i_resp(i_ixy(1, :) == iInd, cInd, cfInd);
            if isempty(i_supLbl)
                resp_resh_2d = reshape(curResp, [i_imgWHs(2, iInd), i_imgWHs(1, iInd)]);
            else
                resp_resh_2d = zeros(i_imgWHs(2, iInd), i_imgWHs(1, iInd));
                for pInd=1:size(resp_resh_2d, 1)*size(resp_resh_2d, 2)
                    resp_resh_2d(pInd) = curResp(i_supLbl(iInd).Lbl2ID(i_supLbl(iInd).label(pInd)));
                end
            end
            resp_resh(:, :, cInd, cfInd) = resp_resh_2d;
        end
    end
    o_resp(iInd).resp = resp_resh;


%     for sInd=1:i_nScales
%         iInd_lin = sub2ind([i_nImgs, i_nScales], iInd, sInd);
%         resp_resh = zeros(i_imgWHs(2, iInd_lin), i_imgWHs(1, iInd_lin), nCls, nClf);
%         parfor (cfInd=1:nClf, 64)
%             for cInd=1:nCls
%                 curResp = i_resp(i_ixy(1, :) == iInd_lin, cInd, cfInd);
%                 if isempty(i_supLbl)
%                     resp_resh_2d = reshape(curResp, [i_imgWHs(2, iInd_lin), i_imgWHs(1, iInd_lin)]);
%                 else
%                     resp_resh_2d = zeros(i_imgWHs(2, iInd_lin), i_imgWHs(1, iInd_lin));
%                     for pInd=1:size(resp_resh_2d, 1)*size(resp_resh_2d, 2)
%                         resp_resh_2d(pInd) = curResp(i_supLbl(iInd_lin).Lbl2ID(i_supLbl(iInd_lin).label(pInd)));
%                     end
%                 end
%                 resp_resh(:, :, cInd, cfInd) = resp_resh_2d;
%             end
%         end
%         o_resp(iInd, sInd).resp = resp_resh;
%     end
end
end

