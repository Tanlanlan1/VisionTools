#include "SemSegUtils.h"

void FitStump_binary(
        struct XMeta &i_x_meta, Mat<double> &i_zs, Mat<double> &i_ws, const mxArray* i_params, //FIXME: i_params.nCls can make bugs...in the future
        JBMdl& o_mdl, Mat<double> &o_hs){
    // init
    int featDim = *((int *)mxGetData(mxGetField(i_params, 0, "featDim")));
//     int nData = *((int *)mxGetData(mxGetField(i_params, 0, "nData")));
    int nData = i_ws.Size(1);
    int nCls_ori = *((int *)mxGetData(mxGetField(i_params, 0, "nCls")));
    int fBinary = *GetIntPnt(mxGetField(i_params, 0, "binary"), 0, 0);
    double featSelRatio = *((double *)mxGetData(mxGetField(i_params, 0, "featSelRatio")));
    vector<double> featValRange;
    for(int i=0; i<mxGetNumberOfElements(mxGetField(i_params, 0, "featValRange")); ++i)
        featValRange.push_back(*GetDblPnt(mxGetField(i_params, 0, "featValRange"), i, 0));
    int nCls = 2;
    
    // fit a stump. Find the best weaklearner given an S
    
    vector<int> S(2); S[0] = 1; // dummy
    vector<int> randFeatInd;
    GetRandPerm(0, featDim-1, randFeatInd);
    randFeatInd.resize(round(featDim*featSelRatio));
    vector<double> kc_hat(nCls); // dummy
    
    double Jwse_S_best = mxGetInf();
    JBMdl mdl_S_best;
    #pragma omp parallel for
    for(int fIndInd=0; fIndInd<randFeatInd.size(); ++fIndInd){
        int fInd = randFeatInd[fIndInd];

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
                if(S[cInd] == 0)
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
            JBMdl mdl_tmp(a_hat, b_hat, fInd, curTheta, kc_hat, S);
            double Jwse_S_f_t = CalcJwse_binary(i_ws, i_zs, xs_tmp, mdl_tmp);

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
    
    // free memory and return
    o_mdl = mdl_S_best;
    vector<double> xs_tmp(nData);
    for(int dInd=0; dInd<nData; ++dInd)
        xs_tmp[dInd] = GetithTextonBoost(dInd, o_mdl.f, i_x_meta);
    Geths(o_hs, nData, nCls, xs_tmp, o_mdl);
}

void FitStumpForAllS(
        struct XMeta &i_x_meta, Mat<double> &i_zs, Mat<double> &i_ws, const mxArray* i_params, //FIXME: i_params.nCls can make bugs...in the future
        JBMdl& o_mdl, Mat<double> &o_hs){
    // init
    int featDim = *((int *)mxGetData(mxGetField(i_params, 0, "featDim")));
//     int nData = *((int *)mxGetData(mxGetField(i_params, 0, "nData")));
    int nData = i_ws.Size(1);
    int nCls_ori = *((int *)mxGetData(mxGetField(i_params, 0, "nCls")));
    int fBinary = *GetIntPnt(mxGetField(i_params, 0, "binary"), 0, 0);
    double featSelRatio = *((double *)mxGetData(mxGetField(i_params, 0, "featSelRatio")));
    vector<double> featValRange;
    for(int i=0; i<mxGetNumberOfElements(mxGetField(i_params, 0, "featValRange")); ++i)
        featValRange.push_back(*GetDblPnt(mxGetField(i_params, 0, "featValRange"), i, 0));
    int nCls;
    if(fBinary) { //FIXME: duplicated
        nCls = 2;
    }
    else{
        nCls = nCls_ori;
    }
    
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

void LearnJointBoost(struct XMeta &i_x_meta, const mxArray* i_ys, const mxArray* i_params, Mat<JBMdl> &o_mdls){
    // get params
    int nWeakLearner = *GetIntPnt(mxGetField(i_params, 0, "nWeakLearner"), 0, 0);
    int nData_ori = *GetIntPnt(mxGetField(i_params, 0, "nData"), 0, 0);
    int nPerClsSample = *GetIntPnt(mxGetField(i_params, 0, "nPerClsSample"), 0, 0);
    int nCls_ori = *GetIntPnt(mxGetField(i_params, 0, "nCls"), 0, 0);
    int verbosity = *GetIntPnt(mxGetField(i_params, 0, "verbosity"), 0, 0);
    int fBinary = *GetIntPnt(mxGetField(i_params, 0, "binary"), 0, 0);
    int learnBG = *GetIntPnt(mxGetField(i_params, 0, "learnBG"), 0, 0);
    int nCls, nRep, nData;
    if(fBinary==1) { //FIXME: duplicated
        nRep = nCls_ori;
        nCls = 2;
        nData = nPerClsSample*2;
        if(learnBG == 0)
            nRep--;
    }
    else{
        nRep = 1;
        nCls = nCls_ori;
        nData = nData_ori;
        // always learn BG
    }
    
    // allocate mdls
    o_mdls.Resize(nWeakLearner, nRep);
    // multiple binary classifiers or one multiclass classifier
    for(int rInd=0; rInd<nRep; ++rInd){
        // allocate zs, ws    
        Mat<double> zs(-1, nData, nCls);    
        Mat<double> ws(1, nData, nCls);
        struct XMeta x_meta = i_x_meta;
        vector<int> ys;
        
        // init weight
        if(fBinary==1){
            // balance weights of labels
            int nPos = nPerClsSample; //FIXME: assumes positive data is smaller
            
            // counts neg/pos
            vector<int> sampleInd;
            vector<int> posInd;
            vector<int> negInd;
            for(int dInd=0; dInd<nData_ori; ++dInd){
                int clsLabel = (int)((*GetIntPnt(i_ys, dInd, 0)) != (rInd+1)) + 1;
                if (clsLabel == 1)
                    // count positive
                    posInd.push_back(dInd);
                else
                    // count negative
                    negInd.push_back(dInd);
            }
            
            // sample
            sampleInd = posInd;
            vector<int> randNegInd;
            GetRandPerm(0, negInd.size()-1, randNegInd);
            for(int i=0; i<nPos; ++i)
                sampleInd.push_back(negInd[randNegInd[i]]);
            assert(sampleInd.size() == nData);
            printf("sampleSize: %d\n", sampleInd.size());
            
            // construct XMeta, ys
            vector<int> ixys_s;
            for(int sInd=0; sInd<sampleInd.size(); ++sInd){
                // ys
                ys.push_back(*GetIntPnt(i_ys, sampleInd[sInd], 0));
                
                // xMeta
                ixys_s.push_back(i_x_meta.ixys[sampleInd[sInd]*3 + 0]);
                ixys_s.push_back(i_x_meta.ixys[sampleInd[sInd]*3 + 1]);
                ixys_s.push_back(i_x_meta.ixys[sampleInd[sInd]*3 + 2]);
            }
            x_meta.ixys = ixys_s;
            
                        
//             for(int dInd=0; dInd<nData; ++dInd){
//                 int clsLabel = (int)((*GetIntPnt(i_ys, dInd, 0)) != (rInd+1)) + 1;
//                 if(clsLabel == 1){
//                     ws.GetRef(dInd, 0) = nCls_ori-1; // zero base
//                     ws.GetRef(dInd, 1) = nCls_ori-1; // zero base
//                 }
//             }   
//             
//             int dInd = -1;
//             while(nNeg!=nPos){
//                 dInd = (dInd+1)%nData;
//                 if(ws.GetRef(dInd, 0) == 0 && ws.GetRef(dInd, 1) == 0)
//                     continue;
//                 
//                 int clsLabel = (int)((*GetIntPnt(i_ys, dInd, 0)) != (rInd+1)) + 1;
//                 
// //                 if(clsLabel == 2 && rand()%(nCls_ori-1) == 0){ // negative
//                 if(clsLabel == 2 && (*GetIntPnt(i_ys, dInd, 0)) != 4){
//                     ws.GetRef(dInd, 0) = 0; // zero base
//                     ws.GetRef(dInd, 1) = 0; // zero base
//                     nNeg--;
//                 }
//             }
//             assert(nPos == nNeg);
        }
        else
        {
            assert(0);//FIXME:
        }
        
        // init labels
        for(int dInd=0; dInd<nData; ++dInd){
            if(ys[dInd] == 0) // bg
                continue;
            int clsLabel;
            if(fBinary==1){ //FIXME: not beautiful
                clsLabel = (int)(ys[dInd] != (rInd+1)) + 1;
            }else{
                clsLabel = ys[dInd];
            }
            zs.GetRef(dInd, clsLabel-1) = 1; // zero base
        }           

        // train weak classifiers
        char buf[1024];
        Mat<double> hs(0, nData, nCls);
        for(int m=0; m<nWeakLearner; ++m){
            if(verbosity>=1){
                sprintf(buf, "* [%dth classifier] boosting iter: %d/%d...", rInd+1, m+1, nWeakLearner);
                cout << buf;
            }

            // fit a stump
            if(fBinary == 1){
                FitStump_binary(x_meta, zs, ws, i_params, o_mdls.GetRef(m, rInd), hs);
                if(verbosity >= 1){
                    sprintf(buf, "J_wse = % 12.06f", CalcJwse_binary(ws, zs, hs));
                    cout << buf << endl;
                    fflush(stdout);
                }
            }else{
                FitStumpForAllS(x_meta, zs, ws, i_params, o_mdls.GetRef(m, rInd), hs);
                if(verbosity >= 1){
                    sprintf(buf, "J_wse = % 12.06f", CalcJwse(ws, zs, hs));
                    cout << buf << endl;
                    fflush(stdout);
                }
            }
            // update ws
            UpdWs(ws, zs, hs);

            
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
    Mat<JBMdl> mdls;
    LearnJointBoost(xMeta, i_ys, i_params, mdls);
    mxArray* o_mdls = 0;
    ConvCMdl2MMdl(mdls, &o_mdls);
    
    // return
    plhs[0] = o_mdls;

    return;
}
