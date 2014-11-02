function [x_meta_mex, label_mex, JBParams_mex] = convParamsType(x_meta, label, JBParams)
x_meta_mex = x_meta;
% x_meta.ixy
x_meta_mex.ixy = int32(x_meta_mex.ixy);
% x_meta.intImgfeat
for iInd=1:numel(x_meta_mex.intImgFeat)
    x_meta_mex.intImgFeat(iInd).TextonIntImg = double(x_meta_mex.intImgFeat(iInd).TextonIntImg);
end
% x_meta.tbParams
x_meta_mex.TBParams.LOFilterWH = int32(x_meta_mex.TBParams.LOFilterWH);
x_meta_mex.TBParams.nTexton = int32(x_meta_mex.TBParams.nTexton);
x_meta_mex.TBParams.parts = int32(x_meta_mex.TBParams.parts);
% label
label_mex = int32(label);
% JBParams
JBParams_mex = JBParams;
JBParams_mex.nWeakLearner = int32(JBParams_mex.nWeakLearner);
JBParams_mex.featDim = int32(JBParams_mex.featDim);
JBParams_mex.nData = int32(JBParams_mex.nData);
JBParams_mex.nPerClsSample = int32(JBParams_mex.nPerClsSample);
JBParams_mex.nCls = int32(JBParams_mex.nCls);
JBParams_mex.binary = int32(JBParams_mex.binary);
JBParams_mex.learnBG = int32(JBParams_mex.learnBG);
JBParams_mex.featSelRatio = double(JBParams_mex.featSelRatio);
JBParams_mex.featValRange = double(JBParams_mex.featValRange(:));
JBParams_mex.verbosity = int32(JBParams_mex.verbosity);


end