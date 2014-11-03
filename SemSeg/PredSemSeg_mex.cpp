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

int NTHREAD_MAX;

void PredJointBoost(Mat<double>& o_Hs, struct XMeta &i_xs_meta, Mat<JBMdl>& i_mdls, const mxArray *i_params){
    // init
    
    int nWeakLearner = *GetIntPnt(mxGetField(i_params, 0, "nWeakLearner"), 0, 0);
    int nData = *GetIntPnt(mxGetField(i_params, 0, "nData"), 0, 0);
    int nCls_ori = *GetIntPnt(mxGetField(i_params, 0, "nCls"), 0, 0);
    int nClsf = i_mdls.Size(2);
    int fBinary = *GetIntPnt(mxGetField(i_params, 0, "binary"), 0, 0);
    int verbosity = *GetIntPnt(mxGetField(i_params, 0, "verbosity"), 0, 0);
    int nCls;
    if(fBinary==1) { //FIXME: duplicated
        nCls = 2-1;
    }
    else{
        nCls = nCls_ori;
    }
    // predict
    int nThread_out = min(nClsf, (int)round(sqrt(NTHREAD_MAX)));
    #pragma omp parallel for num_threads(nThread_out)
    for(int cfInd=0; cfInd<nClsf; ++cfInd){
        if(verbosity>=2 && cfInd%10==0){
            char buf[1024];
            sprintf(buf, "* [%dth classifier] predict...", cfInd+1);
            cout << buf;
            fflush(stdout);
        }
        
        int nThread_in = round(NTHREAD_MAX/nThread_out);
        #pragma omp parallel for num_threads(nThread_in)
        for(int dInd=0; dInd<nData; ++dInd){
            for(int cInd=0; cInd<nCls; ++cInd){    
                double H = 0;
                for(int m=0; m<nWeakLearner; ++m){
                    // x
                    double x = GetithTextonBoost(dInd, i_mdls.GetRef(m, cfInd).f, i_xs_meta);
                    // h
                    double h = Geth(x, cInd, i_mdls.GetRef(m, cfInd));
                    // H
                    H += h;
                }
                o_Hs.GetRef(dInd, cInd, cfInd) = H;
            }
        }
        
        if(verbosity>=2 && cfInd%10==0){
            char buf[1024];
            sprintf(buf, "done");
            cout << buf << endl;
            fflush(stdout);
        }
    }
}

void
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    //openMP
    NTHREAD_MAX = NTHREAD;
    omp_set_dynamic(0);                     // disable dynamic teams
    omp_set_nested(1);
    omp_set_num_threads((int)round(NTHREAD_MAX*0.8)); // override env var OMP_NUM_THREADS
    char buf[1024];
    sprintf(buf, "* #thread: %d/%d", (int)round(NTHREAD_MAX*0.8), (int)NTHREAD_MAX);
    cout << buf << endl;
    fflush(stdout);

    // i_xs
    const mxArray* xs_meta = prhs[0];
    struct XMeta xMeta;
    ConvMXMeta2CXMeta(xs_meta, xMeta);
    // i_mdls
    const mxArray* i_mdls = prhs[1];
    Mat<JBMdl> mdls_cpp;
    ConvMMdl2CMdl(i_mdls, mdls_cpp);

    // struct i_params
    const mxArray* i_params = prhs[2];
    int fBinary = *GetIntPnt(mxGetField(i_params, 0, "binary"), 0, 0);
    int nData = *GetIntPnt(mxGetField(i_params, 0, "nData"), 0, 0);
    int nClsf = mdls_cpp.Size(2);
    int nCls_ori = *GetIntPnt(mxGetField(i_params, 0, "nCls"), 0, 0);
    int nCls;
    if(fBinary==1) { //FIXME: duplicated
        nCls = 2;
    }
    else{
        nCls = nCls_ori;
    }
    
    // learn
    Mat<double> Hs(0, nData, nCls, nClsf);
    PredJointBoost(Hs, xMeta, mdls_cpp, i_params);
    
    // reshape result
//     Mat<double> Hs_resh(
            
    // return
    mwSize sz[3];
    sz[0] = nData;
    sz[1] = nCls;
    sz[2] = nClsf;
    mxArray* dist = mxCreateNumericArray(3, sz, mxDOUBLE_CLASS, mxREAL);
    ConvMat2MMat(Hs, dist);
    
    plhs[0] = dist;

    return;
}
