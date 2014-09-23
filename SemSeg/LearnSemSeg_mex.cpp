#include "SemSegUtils.h"

void FitStumpForAllS(
        struct XMeta &i_x_meta, const mxArray* i_zs, const mxArray* i_ws, const mxArray* i_params, const int i_mInd,
        mxArray* o_mdls, mxArray* o_hs){
    // init
    int featDim = *((int *)mxGetData(mxGetField(i_params, 0, "featDim")));
    int nData = *((int *)mxGetData(mxGetField(i_params, 0, "nData")));
    int nCls = *((int *)mxGetData(mxGetField(i_params, 0, "nCls")));
    double featSelRatio = *((double *)mxGetData(mxGetField(i_params, 0, "featSelRatio")));
    vector<double> featValRange;
    for(int i=0; i<mxGetNumberOfElements(mxGetField(i_params, 0, "featValRange")); ++i)
        featValRange.push_back(*GetDblPnt(mxGetField(i_params, 0, "featValRange"), i, 0));
    
    mxArray* xs_tmp = mxCreateDoubleMatrix(nData, 1, mxREAL);
    
    // n* = argmin_n Jwse(n)
    double Jwse_best = mxGetInf();
    mxArray* mdl_best = InitMdl(1, nCls);
    mxArray* hs_best = mxCreateDoubleMatrix(nData, nCls, mxREAL);
    mxArray* mdl_S_best = InitMdl(1, nCls);
    mxArray* hs_S_best = mxCreateDoubleMatrix(nData, nCls, mxREAL);;
    
    // greedly select S(n) 
    vector<int> S(nCls);
    for(int maxSize=0; maxSize<nCls; ++maxSize){
        for(int candInd=0; candInd<nCls; ++candInd){
            if(S[candInd] == 1)
                continue;
            // choose a candidate S
            vector<int> curS = S;
            curS[candInd] = 1;
            
            // estimate k, which is independent on f and theta 
            // precalc kc
            vector<double> kc_hat(nCls);
            for(int cInd=0; cInd<nCls; ++cInd){
                if(curS[cInd] == 1){
                    kc_hat[cInd] = mxGetNaN();
                    
                }else{
                    double n = 0, dn = 0;
                    for(int dInd=0; dInd<nData; ++dInd){
                        dn += *GetDblPnt(i_ws, dInd, cInd);
                        n += (*GetDblPnt(i_ws, dInd, cInd))*(*GetDblPnt(i_zs, dInd, cInd));
                    }
                    kc_hat[cInd] = n/dn;
                }
            }            
            
            // fit a stump. Find a weak learner given an S
            double Jwse_S_best = mxGetInf();
            
            for(int fInd=0; fInd<featDim; ++fInd){
                if(((double)(rand()%1000))/1000 > featSelRatio)
                    continue;
                
                // precalc xs
                for(int dInd=0; dInd<nData; ++dInd){ 
                    (*GetDblPnt(xs_tmp, dInd, 0)) = GetithTextonBoost_new(dInd, fInd, i_x_meta);
                }
                
                #pragma omp parallel for
                for(int tInd=0; tInd<featValRange.size(); ++tInd){
                    double curTheta = featValRange[tInd];
                    
                    // estimate a and b
                    double n_a = 0, dn_a = 1e-32, n_b = 0, dn_b = 1e-32;
                    for(int cInd=0; cInd<nCls; ++cInd){
                        if(curS[cInd] == 0)
                            continue;
                        for(int dInd=0; dInd<nData; ++dInd){
                            n_a += (*GetDblPnt(i_ws, dInd, cInd))*(*GetDblPnt(i_zs, dInd, cInd))*((*GetDblPnt(xs_tmp, dInd, 0))>curTheta);
                            dn_a += (*GetDblPnt(i_ws, dInd, cInd))*((*GetDblPnt(xs_tmp, dInd, 0))>curTheta);
                            n_b += (*GetDblPnt(i_ws, dInd, cInd))*(*GetDblPnt(i_zs, dInd, cInd))*((*GetDblPnt(xs_tmp, dInd, 0))<=curTheta);
                            dn_b += (*GetDblPnt(i_ws, dInd, cInd))*((*GetDblPnt(xs_tmp, dInd, 0))<=curTheta);
                        }
                    }
                    double a_hat = n_a/dn_a;
                    double b_hat = n_b/dn_b;

                    // calc cost
                    double Jwse_S_f_t = CalcJwse_ts(i_ws, i_zs, xs_tmp, a_hat, b_hat, fInd, curTheta, kc_hat, curS);
                    
                    // keep the best and free memory
                    #pragma omp critical
                    {
                    if(Jwse_S_best > Jwse_S_f_t){
                        Jwse_S_best = Jwse_S_f_t;
                        SetMdlField(mdl_S_best, a_hat, b_hat, fInd, curTheta, kc_hat, curS);
                    }
                    }
                }
            }

            // keep the best and free memory
            if(Jwse_best > Jwse_S_best){
                Jwse_best = Jwse_S_best;
                CopyMdlVal(mdl_best, 0, mdl_S_best, 0);
                S = curS;
            }
            
        }
    }
    
    // free memory and return
    CopyMdlVal(o_mdls, i_mInd, mdl_best, 0);
    
    for(int dInd=0; dInd<nData; ++dInd)
        (*GetDblPnt(xs_tmp, dInd, 0)) = GetithTextonBoost_new(dInd, (*GetIntPnt(mxGetField(o_mdls, i_mInd, "f"), 0, 0)), i_x_meta);
    Geths(o_hs, nData, nCls, xs_tmp, o_mdls, i_mInd);
    
    mxDestroyArray(mdl_best);
    mxDestroyArray(hs_best);
    mxDestroyArray(mdl_S_best);
    mxDestroyArray(hs_S_best);
    mxDestroyArray(xs_tmp);
}

mxArray* LearnJointBoost(struct XMeta &i_x_meta, const mxArray* i_ys, const mxArray* i_params){
    // get params
    int nWeakLearner = *GetIntPnt(mxGetField(i_params, 0, "nWeakLearner"), 0, 0);
    int nData = *GetIntPnt(mxGetField(i_params, 0, "nData"), 0, 0);
    int nCls = *GetIntPnt(mxGetField(i_params, 0, "nCls"), 0, 0);
    int verbosity = *GetIntPnt(mxGetField(i_params, 0, "verbosity"), 0, 0);
    
    // allocate zs, ws
    mxArray* zs = mxCreateDoubleMatrix(nData, nCls, mxREAL);
    SetDblVal(zs, -1);
    mxArray* ws = mxCreateDoubleMatrix(nData, nCls, mxREAL);
    SetDblVal(ws, 1);

    // allocate mdls
    mxArray* mdls = InitMdl(nWeakLearner, nCls);
    
    
    // init labels
    for(int dInd=0; dInd<nData; ++dInd){
        if((*GetIntPnt(i_ys, dInd, 0)) == 0) // bg
            continue;
        (*GetDblPnt(zs, dInd, (*GetIntPnt(i_ys, dInd, 0))-1)) = 1; // zero base
    }

    // train weak classifiers
    char buf[1024];
    mxArray *hs = mxCreateDoubleMatrix(nData, nCls, mxREAL);
    for(int m=0; m<nWeakLearner; ++m){
        if(verbosity>=1){
            sprintf(buf, "* boosting iter: %d/%d...", m+1, nWeakLearner);
            cout << buf;
        }
        
        // fit a stump
        FitStumpForAllS(i_x_meta, zs, ws, i_params, m, mdls, hs);
        // update ws
        UpdWs(ws, zs, hs);
    
        if(verbosity >= 1){
            sprintf(buf, "J_wse = % 12.06f", CalcJwse(ws, zs, hs));
            cout << buf << endl;
            fflush(stdout);
        }
    }
    
    // free memory
    mxDestroyArray(zs);
    mxDestroyArray(ws);
    mxDestroyArray(hs);
    
    // return
    return mdls;
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
    // double* i_ys
    const mxArray* i_ys = prhs[1];
    // struct i_params
    const mxArray* i_params = prhs[2];
    // learn
    mxArray* mdls = LearnJointBoost(xMeta, i_ys, i_params);
    // return
    plhs[0] = mdls;

    return;
}
