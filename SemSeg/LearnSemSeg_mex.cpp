#include "SemSegUtils.h"

void FitStumpForAllS(
        struct XMeta &i_x_meta, Mat &i_zs, Mat &i_ws, const mxArray* i_params, 
        JBMdl& o_mdl, Mat &o_hs){
    // init
    int featDim = *((int *)mxGetData(mxGetField(i_params, 0, "featDim")));
    int nData = *((int *)mxGetData(mxGetField(i_params, 0, "nData")));
    int nCls = *((int *)mxGetData(mxGetField(i_params, 0, "nCls")));
    double featSelRatio = *((double *)mxGetData(mxGetField(i_params, 0, "featSelRatio")));
    vector<double> featValRange;
    for(int i=0; i<mxGetNumberOfElements(mxGetField(i_params, 0, "featValRange")); ++i)
        featValRange.push_back(*GetDblPnt(mxGetField(i_params, 0, "featValRange"), i, 0));
    
    
    // n* = argmin_n Jwse(n)
    double Jwse_best = mxGetInf();
    JBMdl mdl_best(nCls);
    JBMdl mdl_S_best;
    
    // greedly select S(n) 
    for(int maxSize=0; maxSize<nCls; ++maxSize){
        for(int candInd=0; candInd<nCls; ++candInd){
            if(mdl_best.S[candInd] == 1)
                continue;
            // choose a candidate S
            vector<int> curS = mdl_best.S;
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
                        dn += i_ws.GetRef(dInd, cInd);
                        n += i_ws.GetRef(dInd, cInd) * i_zs.GetRef(dInd, cInd);
                    }
                    kc_hat[cInd] = n/dn;
                }
            }            
            
            // fit a stump. Find a weak learner given an S
            vector<int> randFeatInd;
            GetRandPerm(0, featDim-1, randFeatInd);
            randFeatInd.resize(round(featDim*featSelRatio));
            double Jwse_S_best = mxGetInf();
            #pragma omp parallel for
            for(int fIndInd=0; fIndInd<randFeatInd.size(); ++fIndInd){
                int fInd = randFeatInd[fIndInd];
//             for(int fInd=0; fInd<featDim; ++fInd){
//                 if(((double)(rand()%1000))/1000 > featSelRatio)
//                     continue;
                
                // precalc xs
                vector<double> xs_tmp(nData);
                for(int dInd=0; dInd<nData; ++dInd){ 
                    xs_tmp[dInd] = GetithTextonBoost(dInd, fInd, i_x_meta);
                }
                
                for(int tInd=0; tInd<featValRange.size(); ++tInd){
                    double curTheta = featValRange[tInd];
                    
                    // estimate a and b
                    double n_a = 0, dn_a = 1e-32, n_b = 0, dn_b = 1e-32;
                    for(int cInd=0; cInd<nCls; ++cInd){
                        if(curS[cInd] == 0)
                            continue;
                        for(int dInd=0; dInd<nData; ++dInd){
                            n_a += i_ws.GetRef(dInd, cInd) * i_zs.GetRef(dInd, cInd) * (xs_tmp[dInd]>curTheta);
                            dn_a += i_ws.GetRef(dInd, cInd) * (xs_tmp[dInd]>curTheta);
                            n_b += i_ws.GetRef(dInd, cInd) * i_zs.GetRef(dInd, cInd) * (xs_tmp[dInd]<=curTheta);
                            dn_b += i_ws.GetRef(dInd, cInd) * (xs_tmp[dInd]<=curTheta);
                        }
                    }
                    double a_hat = n_a/dn_a;
                    double b_hat = n_b/dn_b;

                    // calc cost
                    JBMdl mdl_tmp(a_hat, b_hat, fInd, curTheta, kc_hat, curS);
                    double Jwse_S_f_t = CalcJwse(i_ws, i_zs, xs_tmp, mdl_tmp);
                    
                    // keep the best and free memory
                    #pragma omp critical
                    {
                    if(Jwse_S_best > Jwse_S_f_t){
                        Jwse_S_best = Jwse_S_f_t;
                        mdl_S_best = mdl_tmp;
                    }
                    }
                }
            }

            // keep the best and free memory
            if(Jwse_best > Jwse_S_best){
                Jwse_best = Jwse_S_best;
                mdl_best = mdl_S_best;
            }   
        }
    }
    
    // free memory and return
    o_mdl = mdl_best;
    vector<double> xs_tmp(nData);
    for(int dInd=0; dInd<nData; ++dInd)
        xs_tmp[dInd] = GetithTextonBoost(dInd, o_mdl.f, i_x_meta);
    Geths(o_hs, nData, nCls, xs_tmp, o_mdl);
    
}

void LearnJointBoost(struct XMeta &i_x_meta, const mxArray* i_ys, const mxArray* i_params, vector<JBMdl> &o_mdls){
    // get params
    int nWeakLearner = *GetIntPnt(mxGetField(i_params, 0, "nWeakLearner"), 0, 0);
    int nData = *GetIntPnt(mxGetField(i_params, 0, "nData"), 0, 0);
    int nCls = *GetIntPnt(mxGetField(i_params, 0, "nCls"), 0, 0);
    int verbosity = *GetIntPnt(mxGetField(i_params, 0, "verbosity"), 0, 0);
    
    // allocate zs, ws    
    Mat zs(-1, nData, nCls);    
    Mat ws(1, nData, nCls);

    // allocate mdls
//     mxArray* mdls = InitMdl(nWeakLearner, nCls);
    o_mdls.resize(nWeakLearner);
    // init labels
    for(int dInd=0; dInd<nData; ++dInd){
        if((*GetIntPnt(i_ys, dInd, 0)) == 0) // bg
            continue;
        zs.GetRef(dInd, (*GetIntPnt(i_ys, dInd, 0))-1) = 1; // zero base
    }

    // train weak classifiers
    char buf[1024];
    Mat hs(0, nData, nCls);
    for(int m=0; m<nWeakLearner; ++m){
        if(verbosity>=1){
            sprintf(buf, "* boosting iter: %d/%d...", m+1, nWeakLearner);
            cout << buf;
        }
        
        // fit a stump
        FitStumpForAllS(i_x_meta, zs, ws, i_params, o_mdls[m], hs);
        // update ws
        UpdWs(ws, zs, hs);
    
        if(verbosity >= 1){
            sprintf(buf, "J_wse = % 12.06f", CalcJwse(ws, zs, hs));
            cout << buf << endl;
            fflush(stdout);
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
    // double* i_ys
    const mxArray* i_ys = prhs[1];
    // struct i_params
    const mxArray* i_params = prhs[2];
    // learn
    int nWeakLearner = *GetIntPnt(mxGetField(i_params, 0, "nWeakLearner"), 0, 0);
    int nCls = *GetIntPnt(mxGetField(i_params, 0, "nCls"), 0, 0);
    vector<JBMdl> mdls;
    mxArray* o_mdls = InitMdl(nWeakLearner, nCls);
    
    LearnJointBoost(xMeta, i_ys, i_params, mdls);
    ConvCMdl2MMdl(mdls, o_mdls);
    
    // return
    plhs[0] = o_mdls;

    return;
}
