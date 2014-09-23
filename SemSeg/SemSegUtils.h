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
    int nParts = mxGetN(mxGetField(TBParams, 0, "parts"));
    
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
    
    // extract
    
    mxArray *curIntImg = mxGetField(intImgFeat, ixy[0], "feat");
    double partArea = (xy_part_br[0] - xy_part_tl[0] + 1)*(xy_part_br[1] - xy_part_tl[1] + 1);
    double I1 = *GetDblPnt(curIntImg, xy_part_br[1]+1, xy_part_br[0]+1, tInd); // one base due to the convention of an integral image
    double I2 = *GetDblPnt(curIntImg, xy_part_tl[1]  , xy_part_br[0]+1, tInd); // one base due to the convention of an integral image
    double I3 = *GetDblPnt(curIntImg, xy_part_br[1]+1, xy_part_tl[0]  , tInd); // one base due to the convention of an integral image
    double I4 = *GetDblPnt(curIntImg, xy_part_tl[1]  , xy_part_tl[0]  , tInd); // one base due to the convention of an integral image
    double val = (I1 - I2 - I3 + I4)/partArea;
    
    // return
    return val;
}

#endif