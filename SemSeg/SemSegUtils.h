#ifndef SEMSEG_UTILS_H
#define SEMSEG_UTILS_H

#include "mex.h"
#include "matrix.h"
#include <string.h>
#include <vector>
#include <math.h>
#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <assert.h>
#include <omp.h>

#define NTHREAD 32

using namespace std;

double* GetDblPnt(const mxArray* i_m, int i_r, int i_c, int i_d){
    const mwSize *pDims = mxGetDimensions(i_m);
    size_t nRows = pDims[0];
    size_t nCols = pDims[1];
    
    double *data = (double*)mxGetData(i_m);
    double *ret = &data[i_r + i_c*nRows + i_d*nRows*nCols];
    return ret;
}


double* GetDblPnt(const mxArray* i_m, int i_r, int i_c){
    size_t nRows = mxGetM(i_m);
    double *data = (double*)mxGetData(i_m);
    double *ret = &data[i_r + i_c*nRows];
    return ret;
}

int* GetIntPnt(const mxArray* i_m, int i_r, int i_c){
    size_t nRows = mxGetM(i_m);
    int *data = (int*)mxGetData(i_m);
    int *ret = &data[i_r + i_c*nRows];
    return ret;
}

void SetDblVal(mxArray* io_m, double i_val)
{
    size_t nRows = mxGetM(io_m);
    size_t nCols = mxGetN(io_m);
    for(int r=0; r<nRows; ++r)
        for(int c=0; c<nCols; ++c)
            (*GetDblPnt(io_m, r, c)) = i_val;
}

void UpdWs(mxArray* io_ws, mxArray* i_zs, mxArray* i_hs){
    // ws = ws.*exp(-zs.*hs);
    size_t nRows = mxGetM(io_ws);
    size_t nCols = mxGetN(io_ws);
    for(int r=0; r<nRows; ++r)
        for(int c=0; c<nCols; ++c)
            (*GetDblPnt(io_ws, r, c)) = // ws
                    (*GetDblPnt(io_ws, r, c))* // ws
                    exp(-(*GetDblPnt(i_zs, r, c))*(*GetDblPnt(i_hs, r, c))); //exp(-zs.*hs)
}

struct JBMdl{
    double a;
    double b;
    int f;
    double theta;
    vector<double> kc;
    vector<int> S;
};

void ConvMMdl2CMdl(const mxArray* i_mdls, vector<struct JBMdl> &o_mdls){
    int nMdls = mxGetNumberOfElements(i_mdls);
    for(int mInd=0; mInd<nMdls; ++mInd){
        
        struct JBMdl mdl;
        mdl.a = *GetDblPnt(mxGetField(i_mdls, mInd, "a"), 0, 0);
        mdl.b = *GetDblPnt(mxGetField(i_mdls, mInd, "b"), 0, 0);
        mdl.f = *GetIntPnt(mxGetField(i_mdls, mInd, "f"), 0, 0);
        mdl.theta = *GetDblPnt(mxGetField(i_mdls, mInd, "theta"), 0, 0);
        mxArray* kc = mxGetField(i_mdls, mInd, "kc");
        mxArray* S = mxGetField(i_mdls, mInd, "S");
        for(int cInd=0; cInd<mxGetNumberOfElements(kc); ++cInd){
            mdl.kc.push_back(*GetDblPnt(kc, cInd, 0));
            mdl.S.push_back(*GetIntPnt(S, cInd, 0));
        }
                
        o_mdls.push_back(mdl);
    }
}

struct TBParams{
    int LOFilterWH[2];
    int nTextons;
    vector<double> parts; // 4xnPart matrix
};

void ConvMTBParams2CTBParams(const mxArray* i_TBParams, struct TBParams &o_TBParams){
    
    // LOFilterWH
    o_TBParams.LOFilterWH[0] = (*GetIntPnt(mxGetField(i_TBParams, 0, "LOFilterWH"), 0, 0));
    o_TBParams.LOFilterWH[1] = (*GetIntPnt(mxGetField(i_TBParams, 0, "LOFilterWH"), 1, 0));
    
    // nTexton
    o_TBParams.nTextons = (*GetIntPnt(mxGetField(i_TBParams, 0, "nTexton"), 0, 0));
    
    // parts
    mxArray *parts = mxGetField(i_TBParams, 0, "parts");
    int nParts = mxGetN(parts);
    for(int pInd=0; pInd<nParts; ++pInd){
        o_TBParams.parts.push_back((*GetIntPnt(parts, 0, pInd)));
        o_TBParams.parts.push_back((*GetIntPnt(parts, 1, pInd)));
        o_TBParams.parts.push_back((*GetIntPnt(parts, 2, pInd)));
        o_TBParams.parts.push_back((*GetIntPnt(parts, 3, pInd)));
    }
}

struct IntImgFeat{
    vector<double> feat;
    int nRows;
    int nCols;
    int nDeps;
};

void ConvMXIntImgFeat2CIntImgFeat(const mxArray* i_intImgFeat, vector<struct IntImgFeat> &o_intImgFeat){
    int nImgs = mxGetNumberOfElements(i_intImgFeat);
    for(int iInd=0; iInd<nImgs; ++iInd){
        struct IntImgFeat intImgFeat;
        mxArray *curIntImg = mxGetField(i_intImgFeat, iInd, "feat");
        const mwSize *dims = mxGetDimensions(curIntImg);
        int nRows = dims[0];
        int nCols = dims[1];
        int nDeps = dims[2];
        for(int k=0; k<nDeps; ++k)
            for(int j=0; j<nCols; ++j)
                for(int i=0; i<nRows; ++i)
                    intImgFeat.feat.push_back(*GetDblPnt(curIntImg, i, j, k));
        intImgFeat.nRows = nRows;
        intImgFeat.nCols = nCols;
        intImgFeat.nDeps = nDeps;
        
        o_intImgFeat.push_back(intImgFeat);
    }
}

struct XMeta{
    vector<struct IntImgFeat> intImgFeat;
    struct TBParams TBParams;
    vector<int> ixys;
};

void ConvMXMeta2CXMeta(const mxArray* i_x_meta, struct XMeta &o_x_meta){
    
    // intImgFeat
    ConvMXIntImgFeat2CIntImgFeat(mxGetField(i_x_meta, 0, "intImgFeat"), o_x_meta.intImgFeat);
    // TBParams
    ConvMTBParams2CTBParams(mxGetField(i_x_meta, 0, "TBParams"), o_x_meta.TBParams);
    // ixy
    mxArray* ixys = mxGetField(i_x_meta, 0, "ixy");
    for(int j=0; j<mxGetN(ixys); ++j){
        for(int i=0; i<mxGetM(ixys); ++i){
            o_x_meta.ixys.push_back(*GetIntPnt(ixys, i, j));
        }
    }
}

mxArray* InitMdl(int i_n, double i_nCls){
    const char * fnames[] = {"a", "b", "f", "theta", "kc", "S"};
    mxArray *o_mdl = mxCreateStructMatrix(i_n, 1, 6, fnames);
    // set fields
    for(int i=0; i<i_n; ++i){
        mxSetField(o_mdl, i, "a", mxCreateNumericMatrix(1, 1, mxDOUBLE_CLASS, mxREAL));
        mxSetField(o_mdl, i, "b", mxCreateNumericMatrix(1, 1, mxDOUBLE_CLASS, mxREAL));
        mxSetField(o_mdl, i, "f", mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL));
        mxSetField(o_mdl, i, "theta", mxCreateNumericMatrix(1, 1, mxDOUBLE_CLASS, mxREAL));
        mxSetField(o_mdl, i, "kc", mxCreateNumericMatrix(i_nCls, 1, mxDOUBLE_CLASS, mxREAL));
        mxSetField(o_mdl, i, "S", mxCreateNumericMatrix(i_nCls, 1, mxINT32_CLASS, mxREAL));
    }
    return o_mdl;
}

void SetMdlField(mxArray* o_mdl, double a, double b, int f, double theta, vector<double> &kc, vector<int> &S){
    // get
    mxArray* a_f = mxGetField(o_mdl, 0, "a");
    mxArray* b_f = mxGetField(o_mdl, 0, "b");
    mxArray* f_f = mxGetField(o_mdl, 0, "f");
    mxArray* theta_f = mxGetField(o_mdl, 0, "theta");
    mxArray* kc_f = mxGetField(o_mdl, 0, "kc");
    mxArray* S_f = mxGetField(o_mdl, 0, "S");
    // set
    *GetDblPnt(a_f, 0, 0) = a;
    *GetDblPnt(b_f, 0, 0) = b;
    *GetIntPnt(f_f, 0, 0) = f;
    *GetDblPnt(theta_f, 0, 0) = theta;
    for(int i=0; i<kc.size(); ++i)
        *GetDblPnt(kc_f, i, 0) = kc[i];
    for(int i=0; i<S.size(); ++i)
        *GetIntPnt(S_f, i, 0) = S[i];
}


void CopyMdlVal(mxArray *o_mdl, int o_mInd, const mxArray *i_mdl, int i_mInd){
    // get
    mxArray* a_o = mxGetField(o_mdl, o_mInd, "a");
    mxArray* b_o = mxGetField(o_mdl, o_mInd, "b");
    mxArray* f_o = mxGetField(o_mdl, o_mInd, "f");
    mxArray* theta_o = mxGetField(o_mdl, o_mInd, "theta");
    mxArray* kc_o = mxGetField(o_mdl, o_mInd, "kc");
    mxArray* S_o = mxGetField(o_mdl, o_mInd, "S");
    
    mxArray* a_i = mxGetField(i_mdl, i_mInd, "a");
    mxArray* b_i = mxGetField(i_mdl, i_mInd, "b");
    mxArray* f_i = mxGetField(i_mdl, i_mInd, "f");
    mxArray* theta_i = mxGetField(i_mdl, i_mInd, "theta");
    mxArray* kc_i = mxGetField(i_mdl, i_mInd, "kc");
    mxArray* S_i = mxGetField(i_mdl, i_mInd, "S");
    
    // set
    *GetDblPnt(a_o, 0, 0) = *GetDblPnt(a_i, 0, 0);
    *GetDblPnt(b_o, 0, 0) = *GetDblPnt(b_i, 0, 0);
    *GetIntPnt(f_o, 0, 0) = *GetIntPnt(f_i, 0, 0);
    *GetDblPnt(theta_o, 0, 0) = *GetDblPnt(theta_i, 0, 0);
    for(int cInd=0; cInd<mxGetM(kc_o); ++cInd)
        *GetDblPnt(kc_o, cInd, 0) = *GetDblPnt(kc_i, cInd, 0);
    for(int cInd=0; cInd<mxGetM(S_o); ++cInd)
        *GetIntPnt(S_o, cInd, 0) = *GetIntPnt(S_i, cInd, 0);
}

void CopyhsVal(mxArray *o_hs, mxArray *i_hs){
    for(int cInd=0; cInd<mxGetN(o_hs); ++cInd){
        for(int dInd=0; dInd<mxGetM(o_hs); ++dInd){
            *GetDblPnt(o_hs, dInd, cInd) = *GetDblPnt(i_hs, dInd, cInd);
        }
    }
}

double Geth(double x, double a, double b, double theta, double kc, int S){
    // calc h
    double h;
    if(S == 1){
        if(x>theta)
            h = a;
        else
            h = b;
    }else{
        h = kc;
    }
    
    // return
    return h;
}

void Geths(mxArray *o_hs, int i_nData, int i_nCls, const mxArray *i_xs, const mxArray *i_mdl, int i_mInd){
    mxArray* a_f = mxGetField(i_mdl, i_mInd, "a");
    mxArray* b_f = mxGetField(i_mdl, i_mInd, "b");
    mxArray* theta_f = mxGetField(i_mdl, i_mInd, "theta");
    mxArray* kc_f = mxGetField(i_mdl, i_mInd, "kc");
    mxArray* S_f = mxGetField(i_mdl, i_mInd, "S");
    for(int cInd=0; cInd<i_nCls; ++cInd){
        for(int dInd=0; dInd<i_nData; ++dInd){
            if((*GetIntPnt(S_f, cInd, 0)) == 1){
                if((*GetDblPnt(i_xs, dInd, cInd))>(*GetDblPnt(theta_f, 0, 0)))
                    (*GetDblPnt(o_hs, dInd, cInd)) = (*GetDblPnt(a_f, 0, 0));
                else
                    (*GetDblPnt(o_hs, dInd, cInd)) = (*GetDblPnt(b_f, 0, 0));
                    
            }else{
                (*GetDblPnt(o_hs, dInd, cInd)) = (*GetDblPnt(kc_f, cInd, 0));
            }
        }
    }
}

double CalcJwse(const mxArray* i_ws, const mxArray* i_zs, const mxArray* i_hs){
    // o_cost = sum(sum(i_ws.*(i_zs - i_hs).^2));
    size_t nRows = mxGetM(i_ws);
    size_t nCols = mxGetN(i_ws);
    double ret = 0;
    for(int r=0; r<nRows; ++r)
        for(int c=0; c<nCols; ++c){
            ret += (*GetDblPnt(i_ws, r, c))* //ws
                    pow((*GetDblPnt(i_zs, r, c))-(*GetDblPnt(i_hs, r, c)), 2.0); //(zs-hs).^2
        }
    return ret;
}

double CalcJwse_ts(const mxArray* i_ws, const mxArray* i_zs, const mxArray* i_xs, double a, double b, int f, double theta, vector<double> &kc, vector<int> &S){
    // o_cost = sum(sum(i_ws.*(i_zs - i_hs).^2));
    size_t nRows = mxGetM(i_ws);
    size_t nCols = mxGetN(i_ws);
    double ret = 0;
    for(int dInd=0; dInd<nRows; ++dInd)
        for(int cInd=0; cInd<nCols; ++cInd){
            // calc h
            double h = Geth((*GetDblPnt(i_xs, dInd, cInd)), a, b, theta, kc[cInd], S[cInd]);
//             double h;
//             if(S[cInd] == 1){
//                 if((*GetDblPnt(i_xs, dInd, cInd))>theta)
//                     h = a;
//                 else
//                     h = b;
//             }else{
//                 h = kc[cInd];
//             }
            
            // i_ws.*(i_zs - i_hs).^2
            ret += (*GetDblPnt(i_ws, dInd, cInd))* //ws
                    pow((*GetDblPnt(i_zs, dInd, cInd))-h, 2.0); //(zs-hs).^2
        }
    return ret;
}


double GetithTextonBoost(int dInd, int fInd, const mxArray* i_x_meta){
    //i_tbParams.parts(:, i)      ith rectangle in the form of [xmin; xmax; ymin; ymax] 
    //x_meta = struct('ixy', int32(ixy), 'intImgFeat', double(feat), 'TBParams', TBParams);       
    
    // init
    mxArray* intImgFeat = mxGetField(i_x_meta, 0, "intImgFeat");
    mxArray* ixys = mxGetField(i_x_meta, 0, "ixy");
    mxArray* TBParams = mxGetField(i_x_meta, 0, "TBParams");
    
    int LOFWH[2];
    LOFWH[0] = (*GetIntPnt(mxGetField(TBParams, 0, "LOFilterWH"), 0, 0));
    LOFWH[1] = (*GetIntPnt(mxGetField(TBParams, 0, "LOFilterWH"), 1, 0));
    
    int nTextons = (*GetIntPnt(mxGetField(TBParams, 0, "nTexton"), 0, 0));
    
    int pInd = (int)floor(((double)fInd)/((double)nTextons)); // zero-base
    int tInd = fInd - pInd*nTextons; // zero-base
    int part[4];
    part[0] = (*GetIntPnt(mxGetField(TBParams, 0, "parts"), 0, pInd)) - 1; // zero-base 
    part[1] = (*GetIntPnt(mxGetField(TBParams, 0, "parts"), 1, pInd)) - 1; // zero-base
    part[2] = (*GetIntPnt(mxGetField(TBParams, 0, "parts"), 2, pInd)) - 1; // zero-base
    part[3] = (*GetIntPnt(mxGetField(TBParams, 0, "parts"), 3, pInd)) - 1; // zero-base
    
    int ixy[3];
    ixy[0] = (*GetIntPnt(ixys, 0, dInd))-1; // zero-base
    ixy[1] = (*GetIntPnt(ixys, 1, dInd))-1; // zero-base
    ixy[2] = (*GetIntPnt(ixys, 2, dInd))-1; // zero-base
    
    int LOF_tl[2];
    LOF_tl[0] = ixy[1] - ((int)(LOFWH[0]-1)/2);
    LOF_tl[1] = ixy[2] - ((int)(LOFWH[1]-1)/2);
            
    int xy_part_tl[2];
    xy_part_tl[0] = LOF_tl[0] + part[0];
    xy_part_tl[1] = LOF_tl[1] + part[2];
    
    int xy_part_br[2];
    xy_part_br[0] = LOF_tl[0] + part[1];
    xy_part_br[1] = LOF_tl[1] + part[3];
    
    // extract and return
    mxArray *curIntImg = mxGetField(intImgFeat, ixy[0], "feat");
    return (
            *GetDblPnt(curIntImg, xy_part_br[1]+1, xy_part_br[0]+1, tInd) - 
            *GetDblPnt(curIntImg, xy_part_tl[1]  , xy_part_br[0]+1, tInd) - 
            *GetDblPnt(curIntImg, xy_part_br[1]+1, xy_part_tl[0]  , tInd) + 
            *GetDblPnt(curIntImg, xy_part_tl[1]  , xy_part_tl[0]  , tInd))/
            ((xy_part_br[0] - xy_part_tl[0] + 1)*(xy_part_br[1] - xy_part_tl[1] + 1));
}

double GetithTextonBoost_new(int dInd, int fInd, struct XMeta &i_x_meta){
    //i_tbParams.parts(:, i)      ith rectangle in the form of [xmin; xmax; ymin; ymax] 
    //x_meta = struct('ixy', int32(ixy), 'intImgFeat', double(feat), 'TBParams', TBParams);       
    
    // init
    int LOFWH[2];
    LOFWH[0] = i_x_meta.TBParams.LOFilterWH[0];
    LOFWH[1] = i_x_meta.TBParams.LOFilterWH[1];
    
    int nTextons = i_x_meta.TBParams.nTextons;
    
    int pInd = (int)floor(((double)fInd)/((double)nTextons)); // zero-base
    int tInd = fInd - pInd*nTextons; // zero-base
    int part[4];
    part[0] = i_x_meta.TBParams.parts[0 + 4*pInd] - 1; // zero-base 
    part[1] = i_x_meta.TBParams.parts[1 + 4*pInd] - 1; // zero-base
    part[2] = i_x_meta.TBParams.parts[2 + 4*pInd] - 1; // zero-base
    part[3] = i_x_meta.TBParams.parts[3 + 4*pInd] - 1; // zero-base
    
    int ixy[3];
    ixy[0] = i_x_meta.ixys[0 + 3*dInd] - 1; // zero-base
    ixy[1] = i_x_meta.ixys[1 + 3*dInd] - 1; // zero-base
    ixy[2] = i_x_meta.ixys[2 + 3*dInd] - 1; // zero-base
    
    int LOF_tl[2];
    LOF_tl[0] = ixy[1] - ((int)(LOFWH[0]-1)/2);
    LOF_tl[1] = ixy[2] - ((int)(LOFWH[1]-1)/2);
            
    int xy_part_tl[2];
    xy_part_tl[0] = LOF_tl[0] + part[0];
    xy_part_tl[1] = LOF_tl[1] + part[2];
    
    int xy_part_br[2];
    xy_part_br[0] = LOF_tl[0] + part[1];
    xy_part_br[1] = LOF_tl[1] + part[3];
    
    // extract and return
    int nIntImgRows = i_x_meta.intImgFeat[ixy[0]].nRows;
    int nIntImgCols = i_x_meta.intImgFeat[ixy[0]].nCols;
    double I1 = i_x_meta.intImgFeat[ixy[0]].feat[(xy_part_br[1]+1) + nIntImgRows*(xy_part_br[0]+1) + nIntImgRows*nIntImgCols*tInd];
    double I2 = i_x_meta.intImgFeat[ixy[0]].feat[(xy_part_tl[1]) + nIntImgRows*(xy_part_br[0]+1) + nIntImgRows*nIntImgCols*tInd];
    double I3 = i_x_meta.intImgFeat[ixy[0]].feat[(xy_part_br[1]+1) + nIntImgRows*(xy_part_tl[0]) + nIntImgRows*nIntImgCols*tInd];
    double I4 = i_x_meta.intImgFeat[ixy[0]].feat[(xy_part_tl[1]) + nIntImgRows*(xy_part_tl[0]) + nIntImgRows*nIntImgCols*tInd];
    double Area = (xy_part_br[0] - xy_part_tl[0] + 1)*(xy_part_br[1] - xy_part_tl[1] + 1);
    return (I1-I2-I3+I4)/Area;
    
    
//     mxArray *curIntImg = mxGetField(i_x_meta.intImgFeat, ixy[0], "feat");
//     return (
//             *GetDblPnt(curIntImg, xy_part_br[1]+1, xy_part_br[0]+1, tInd) - 
//             *GetDblPnt(curIntImg, xy_part_tl[1]  , xy_part_br[0]+1, tInd) - 
//             *GetDblPnt(curIntImg, xy_part_br[1]+1, xy_part_tl[0]  , tInd) + 
//             *GetDblPnt(curIntImg, xy_part_tl[1]  , xy_part_tl[0]  , tInd))/
//             ((xy_part_br[0] - xy_part_tl[0] + 1)*(xy_part_br[1] - xy_part_tl[1] + 1));
}


#endif