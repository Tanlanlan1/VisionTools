#include "SemSegUtils.h"

mxArray* PredJointBoost(const mxArray* i_xs_meta, const mxArray *i_mdls, const mxArray *i_params){
    // init
    int nWeakLearner = *GetIntPnt(mxGetField(i_params, 0, "nWeakLearner"), 0, 0);
    int nData = *GetIntPnt(mxGetField(i_params, 0, "nData"), 0, 0);
    int nCls = *GetIntPnt(mxGetField(i_params, 0, "nCls"), 0, 0);

    // predict
    mxArray* Hs = mxCreateDoubleMatrix(nData, nCls, mxREAL);
//     #pragma omp parallel for shared(Hs, i_xs_meta, i_mdls, i_params, nWeakLearner, nData, nCls)
    for(int cInd=0; cInd<nCls; ++cInd){    
//         #pragma omp parallel for
        for(int dInd=0; dInd<nData; ++dInd){
        
            double H = 0;
            for(int m=0; m<nWeakLearner; ++m){
                // params
                double a = *GetDblPnt(mxGetField(i_mdls, m, "a"), 0, 0);
                double b = *GetDblPnt(mxGetField(i_mdls, m, "b"), 0, 0);
                int f = *GetIntPnt(mxGetField(i_mdls, m, "f"), 0, 0);
                double theta = *GetDblPnt(mxGetField(i_mdls, m, "theta"), 0, 0);
                mxArray* kc = mxGetField(i_mdls, m, "kc");
                mxArray* S = mxGetField(i_mdls, m, "S");
                // x
                double x = GetithTextonBoost(dInd, f, i_xs_meta);
                // h
                double h = Geth(x, a, b, theta, *GetDblPnt(kc, cInd, 0), *GetIntPnt(S, cInd, 0));
                // H
                H += h;
            }
            (*GetDblPnt(Hs, dInd, cInd)) = H;
        }
    }
    
    // return
    return Hs;
}


void
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    //openMP
    omp_set_dynamic(0);                     // disable dynamic teams
    omp_set_num_threads(NTHREAD); // override env var OMP_NUM_THREADS
    
    // i_xs
    const mxArray* i_xs_meta = prhs[0];
    // i_mdls
    const mxArray* i_mdls = prhs[1];
    // struct i_params
    const mxArray* i_params = prhs[2];
    // learn
    mxArray* dist = PredJointBoost(i_xs_meta, i_mdls, i_params);
    // return
    plhs[0] = dist;

    return;
}
