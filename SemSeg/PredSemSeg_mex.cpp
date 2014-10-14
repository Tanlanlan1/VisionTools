#include "SemSegUtils.h"

// mxArray* PredJointBoost(struct XMeta &i_xs_meta, const vector<struct JBMdl>& i_mdls, const mxArray *i_params){
//     // init
//     int nWeakLearner = *GetIntPnt(mxGetField(i_params, 0, "nWeakLearner"), 0, 0);
//     int nData = *GetIntPnt(mxGetField(i_params, 0, "nData"), 0, 0);
//     int nCls = *GetIntPnt(mxGetField(i_params, 0, "nCls"), 0, 0);
// 
//     // predict
//     mxArray* Hs = mxCreateDoubleMatrix(nData, nCls, mxREAL);
//     #pragma omp parallel for
//     for(int dInd=0; dInd<nData; ++dInd){
//         for(int cInd=0; cInd<nCls; ++cInd){    
//             double H = 0;
//             for(int m=0; m<nWeakLearner; ++m){
//                 // x
//                 double x = GetithTextonBoost(dInd, i_mdls[m].f, i_xs_meta);
//                 // h
//                 double h = Geth(x, cInd, i_mdls[m]);
//                 // H
//                 H += h;
//             }
//             (*GetDblPnt(Hs, dInd, cInd)) = H; //FIXME: matlab funciton involved. can be slow
//         }
//     }
//     
//     // return
//     return Hs;
// }

void PredJointBoost(Mat& o_Hs, struct XMeta &i_xs_meta, const vector<struct JBMdl>& i_mdls, const mxArray *i_params){
    // init
    int nWeakLearner = *GetIntPnt(mxGetField(i_params, 0, "nWeakLearner"), 0, 0);
    int nData = *GetIntPnt(mxGetField(i_params, 0, "nData"), 0, 0);
    int nCls = *GetIntPnt(mxGetField(i_params, 0, "nCls"), 0, 0);

    // predict
    #pragma omp parallel for
    for(int dInd=0; dInd<nData; ++dInd){
        for(int cInd=0; cInd<nCls; ++cInd){    
            double H = 0;
            for(int m=0; m<nWeakLearner; ++m){
                // x
                double x = GetithTextonBoost(dInd, i_mdls[m].f, i_xs_meta);
                // h
                double h = Geth(x, cInd, i_mdls[m]);
                // H
                H += h;
            }
            o_Hs.GetRef(dInd, cInd) = H;
        }
    }
}

void
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    //openMP
    omp_set_dynamic(0);                     // disable dynamic teams
    omp_set_num_threads(NTHREAD); // override env var OMP_NUM_THREADS
    
    // i_xs
    const mxArray* xs_meta = prhs[0];
    struct XMeta xMeta;
    ConvMXMeta2CXMeta(xs_meta, xMeta);
    // i_mdls
    const mxArray* i_mdls = prhs[1];
    vector<struct JBMdl> mdls_cpp;
    ConvMMdl2CMdl(i_mdls, mdls_cpp);
    // struct i_params
    const mxArray* i_params = prhs[2];
    int nData = *GetIntPnt(mxGetField(i_params, 0, "nData"), 0, 0);
    int nCls = *GetIntPnt(mxGetField(i_params, 0, "nCls"), 0, 0);
    
    // learn
    Mat Hs(0, nData, nCls);
    PredJointBoost(Hs, xMeta, mdls_cpp, i_params);
    mxArray* dist = mxCreateDoubleMatrix(nData, nCls, mxREAL);
    ConvMat2MMat(Hs, dist);
    // return
    plhs[0] = dist;

    return;
}
