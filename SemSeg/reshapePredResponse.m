function [o_resp] = reshapePredResponse(i_nImgs, i_nScales, i_ixy, i_imgWHs, i_resp, i_supLbl)

respSt = struct('resp', zeros(0, 0, 0, 0));
o_resp = repmat(respSt, i_nImgs, i_nScales);
coder.varsize('o_resp(:).resp');


nCls = size(i_resp, 2);
nClf = size(i_resp, 3);
for iInd=1:i_nImgs
    for sInd=1:i_nScales
        iInd_lin = sub2ind([i_nImgs, i_nScales], iInd, sInd);
        resp_resh = zeros(i_imgWHs(2, iInd_lin), i_imgWHs(1, iInd_lin), nCls, nClf);
        parfor (cfInd=1:nClf, 64)
            for cInd=1:nCls
                curResp = i_resp(i_ixy(1, :) == iInd_lin, cInd, cfInd);
                if isempty(i_supLbl)
                    resp_resh_2d = reshape(curResp, [i_imgWHs(2, iInd_lin), i_imgWHs(1, iInd_lin)]);
                else
                    resp_resh_2d = zeros(i_imgWHs(2, iInd_lin), i_imgWHs(1, iInd_lin));
                    for pInd=1:size(resp_resh_2d, 1)*size(resp_resh_2d, 2)
                        resp_resh_2d(pInd) = curResp(i_supLbl(iInd_lin).Lbl2ID(i_supLbl(iInd_lin).label(pInd)));
                    end
                end
                resp_resh(:, :, cInd, cfInd) = resp_resh_2d;
            end
        end
        o_resp(iInd, sInd).resp = resp_resh;
    end
end
end




% function [o_resp] = reshapePredResponse(refImgInd, i_params_classifier_binary, i_params_mdlRects, i_params_scales, i_imgs, ixy, imgWHs, dist, supLbl)
% 
% % coder.extrinsic('imresize');
% 
% assert(size(i_imgs, 1) == 1); %%FIXME: assume only one image with different scale images
% iInd = 1;
% refScale = i_imgs(refImgInd).scale;
% imgWH_s1 = [size(i_imgs(iInd, refImgInd).img, 2); size(i_imgs(iInd, refImgInd).img, 1)];
% 
% pred = struct(...
%     'dist', zeros(imgWH_s1(2), imgWH_s1(1), size(dist, 2)), ...
%     'cls', zeros(imgWH_s1(2), imgWH_s1(1)), ...
%     'bbs', []);
% coder.varsize('pred(:).bbs');
% 
% pred(size(dist, 3)) = pred;
% for cfInd=1:size(dist, 3) % for all classifiers
% 
% %     dist_max_s = zeros(imgWH_s1(2), imgWH_s1(1), size(dist, 2));
%     bbs = [];
%     for iInd=1:size(i_imgs, 1)
% %         dist_s = zeros(imgWH_s1(2), imgWH_s1(1), size(i_imgs, 2), size(dist, 2));
%         for sInd=1:size(i_imgs, 2)
%             for cInd=1:size(dist, 2)
%                 iInd_lin = sub2ind(size(i_imgs), iInd, sInd);
%                 curDist = dist(ixy(1, :) == iInd_lin, cInd, cfInd);
% 
%                 % superpixel-wise response map to pixel-wise one. assume
%                 % vector index is same as superpixel id
%                 dist_tmp = zeros(imgWHs(2, iInd_lin), imgWHs(1, iInd_lin));
%                 for supInd=1:size(curDist, 1)
%                     dist_tmp(supLbl(iInd_lin).label(supInd).ind) = curDist(supInd);
%                 end
%                 
%                 % set dist_S
% %                 dist_s(:, :, sInd, cInd) = imresize(dist_tmp, [size(dist_s, 1), size(dist_s, 2)]);
% %                 dist_s(:, :, sInd, cInd) = perform_image_resize(dist_tmp, [size(dist_s, 1), size(dist_s, 2)]);
%                 % find bbs
%                 if i_params_classifier_binary == 1 && cInd == 1
%                     curScale = i_params_scales(sInd);
%                     [curBBs_rect, curScore] = GetBBs(i_params_mdlRects(cfInd, 3:4), dist_tmp); %%FIXME: return only one bb
%                     curBBs_rect = curBBs_rect/curScale*refScale;
%                     bbs = [bbs; curBBs_rect(1) curBBs_rect(2) curBBs_rect(1)+curBBs_rect(3)-1 curBBs_rect(2)+curBBs_rect(4)-1 curScore];
%                 end
%             end
%         end
% %         dist_max_s(:, :, :) = squeeze(mean(dist_s, 3)); % dist_max_s(:, :, :) = squeeze(max(dist_s, [], 3));
%     end
%     % pixel: non-max suppression
% %     pred(cfInd).dist = dist_max_s;
% %     [~, pred(cfInd).cls] = max(dist_max_s, [], 3);
%     % bbs: non-max suppression
%     pred(cfInd).bbs = bbs(nms(bbs, 0.5), :);
% end
% 
% %% return 
% o_resp = pred;
% 
% end
% 
% 
% 
% 
% function [o_BBs_rect, o_score] = GetBBs(i_mdlWH, i_respMap)
% 
% bbWH = round(i_mdlWH);
% respMap = i_respMap;
% % % show the response map
% % if verbosity>=2
% %     figure(43125); imagesc(respMap); axis image;
% % end
% 
% % find max
% respMap = padarray(respMap, max(0, [bbWH(2)-size(respMap, 1) bbWH(1)-size(respMap, 2)]), 0, 'pre'); 
% maxResp = conv2(respMap, ones(bbWH(2), bbWH(1))./(bbWH(2)*bbWH(1)), 'valid');
% % % show the max response map
% % if verbosity>=2
% %     figure(56433); imagesc(maxResp); axis image;
% % end
% 
% % return bb
% [C, y_maxs] = max(maxResp, [], 1);
% [maxVal, xc] = max(C, [], 2);
% yc = y_maxs(xc);
% 
% assert(~isempty(xc));
% % o_BBs_rect = [xc-round(bbWH(1)/2) yc-round(bbWH(2)/2) bbWH(1) bbWH(2)];
% o_BBs_rect = [xc yc bbWH(1) bbWH(2)];
% o_score = maxVal;
% 
% end
