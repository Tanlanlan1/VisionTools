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
#include <algorithm>

#define NTHREAD 64

using namespace std;

void GetRandPerm(int i_start, int i_end, vector<int>& o_list){
    for(int i=i_start; i<=i_end; ++i){
        o_list.push_back(i);
    }
    random_shuffle(o_list.begin(), o_list.end());
}

template<class T>
class Mat{ //FIXME: assume three dimensions
private:
    enum MTYPE {INTERNAL, EXTERNAL};
private:
    MTYPE mType_;
    vector<T> elems_;
    T* elems_ext_; //FIXME: should be const...
    int nRows_;
    int nCols_;
    int nDeps_;
    int nMore_;
public:
    Mat(){
        mType_ = INTERNAL;
    }
    Mat(T i_initVal, int i_nRows, int i_nCols, int i_nDeps=1, int i_nMore=1){
        mType_ = INTERNAL;
        nRows_ = i_nRows;
        nCols_ = i_nCols;
        nDeps_ = i_nDeps;
        nMore_ = i_nMore;
        elems_.resize(nRows_*nCols_*nDeps_*i_nMore);
        for(int i=0; i<elems_.size(); ++i)
            elems_[i] = i_initVal;
    }
    Mat(bool i_ext, T *i_buf, int i_nRows, int i_nCols, int i_nDeps=1, int i_nMore=1){
        mType_ = EXTERNAL;
        elems_ext_ = i_buf;
        nRows_ = i_nRows;
        nCols_ = i_nCols;
        nDeps_ = i_nDeps;
        nMore_ = i_nMore;
    }
    
public:
//     T &GetRef(int i_ind){
//         switch(mType_){
//             case INTERNAL:
//                 return elems_[i_ind];
//             case EXTERNAL:
//                 return elems_ext_[i_ind];
//         }
//     }
    
    
    T &GetRef(int i_r, int i_c=0, int i_d=0, int i_m=0){
        T* tmp;
        switch(mType_){
            case INTERNAL:
                return elems_[i_r + i_c*nRows_ + i_d*nRows_*nCols_ + i_m*nRows_*nCols_*nDeps_];
            case EXTERNAL:
                return elems_ext_[i_r + i_c*nRows_ + i_d*nRows_*nCols_ + i_m*nRows_*nCols_*nDeps_];
        }
    }
    T &operator[](int i_ind){
        return GetRef(i_ind);
    }
    
    
    double Size(){
        return nRows_*nCols_*nDeps_*nMore_;
    }
    double Size(int i_ind){
        switch(i_ind){
            case 1:
                return nRows_;
            case 2:
                return nCols_;
            case 3:
                return nDeps_;
            case 4:
                return nMore_;
        }
    }
    void Resize(int i_nr, int i_nc, int i_nd=1, int i_nm=1){
        if(mType_ == EXTERNAL)
            return;
        
        nRows_ = i_nr;
        nCols_ = i_nc;
        nDeps_ = i_nd;
        nMore_ = i_nm;
        elems_.resize(i_nr*i_nc*i_nd*i_nm);
    }
    
    T* Data(){
        switch(mType_){
            case INTERNAL:
                return elems_.data();
            case EXTERNAL:
                return elems_ext_;
        }
    }
    
};

// double* GetDblPnt(const mxArray* i_m, int i_r, int i_c, int i_d){
//     const mwSize *pDims = mxGetDimensions(i_m);
//     size_t nRows = pDims[0];
//     size_t nCols = pDims[1];
//     
//     double *data = (double*)mxGetData(i_m);
//     double *ret = &data[i_r + i_c*nRows + i_d*nRows*nCols];
//     return ret;
// }


double* GetDblPnt(const mxArray* i_m, int i_r, int i_c, int i_d=0){
    const mwSize *pDims = mxGetDimensions(i_m);
    size_t nRows = pDims[0];
    size_t nCols = pDims[1];
    double *data = (double*)mxGetData(i_m);
    double *ret = &data[i_r + i_c*nRows +i_d*nRows*nCols];
    return ret;
}

int* GetIntPnt(const mxArray* i_m, int i_r, int i_c, int i_d=0){
    const mwSize *pDims = mxGetDimensions(i_m);
    size_t nRows = pDims[0];
    size_t nCols = pDims[1];
    int *data = (int*)mxGetData(i_m);
    int *ret = &data[i_r + i_c*nRows + i_d*nRows*nCols];
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

void UpdWs(Mat<double> &io_ws, Mat<double> &i_zs, Mat<double> &i_hs){
    // ws = ws.*exp(-zs.*hs);
    int nRows = io_ws.Size(1);
    int nCols = io_ws.Size(2);
    #pragma omp parallel for
    for(int r=0; r<nRows; ++r)
        for(int c=0; c<nCols; ++c)
            io_ws.GetRef(r, c) = // ws
                    io_ws.GetRef(r, c)* // ws
                    exp(-i_zs.GetRef(r, c)*i_hs.GetRef(r, c)); //exp(-zs.*hs)
}


class JBMdl{
public:
    double a;
    double b;
    int f;
    double theta;
    vector<double> kc;
    vector<int> S;
public:
    JBMdl(){
    }
    JBMdl(int i_nCls){
        kc.resize(i_nCls);
        S.resize(i_nCls);
    }
    JBMdl(double i_a, double i_b, int i_f, double i_theta, vector<double> &i_kc, vector<int> &i_S){
        a = i_a;
        b = i_b;
        f = i_f;
        theta = i_theta;
        kc = i_kc;
        S = i_S;
    }
};
mxArray* InitMdl(int i_n, int i_nClsf, double i_nCls){
    const char * fnames[] = {"a", "b", "f", "theta", "kc", "S"};
    mxArray *o_mdl = mxCreateStructMatrix(i_n, i_nClsf, 6, fnames);
    // set fields
    for(int i=0; i<i_n*i_nClsf; ++i){
        mxSetField(o_mdl, i, "a", mxCreateNumericMatrix(1, 1, mxDOUBLE_CLASS, mxREAL));
        mxSetField(o_mdl, i, "b", mxCreateNumericMatrix(1, 1, mxDOUBLE_CLASS, mxREAL));
        mxSetField(o_mdl, i, "f", mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL));
        mxSetField(o_mdl, i, "theta", mxCreateNumericMatrix(1, 1, mxDOUBLE_CLASS, mxREAL));
        mxSetField(o_mdl, i, "kc", mxCreateNumericMatrix(i_nCls, 1, mxDOUBLE_CLASS, mxREAL));
        mxSetField(o_mdl, i, "S", mxCreateNumericMatrix(i_nCls, 1, mxINT32_CLASS, mxREAL));
    }
    return o_mdl;
}


// void ConvMat2MMat(Mat<double>& i_mat, const mxArray* o_mat){
//     const mwSize *pDims = mxGetDimensions(o_mat);
//     mwSize nDim = mxGetNumberOfDimensions(o_mat);
//     assert(nDim == 3 || nDim == 2);
//     size_t nRows = pDims[0];
//     size_t nCols = pDims[1];
//     size_t nDeps;
//     if(nDim == 2)
//         nDeps = 1;
//     else
//         nDeps = pDims[2];
//         
//     double* mxData = (double*)mxGetData(o_mat);
//     memcpy(mxData, i_mat.Data(), nRows*nCols*nDeps*sizeof(double));
//     
//     
// //     for(int rInd=0; rInd<nRows; ++rInd)
// //         for(int cInd=0; cInd<nCols; ++cInd)
// //             for(int dInd=0; dInd<nDeps; ++dInd)
// //                 (*GetDblPnt(o_mat, rInd, cInd, dInd)) = i_mat.GetRef(rInd, cInd, dInd);
// }

void ConvMMdl2CMdl(const mxArray* i_mdls, Mat<JBMdl> &o_mdls){
    const mwSize *dims = mxGetDimensions(i_mdls);
    o_mdls.Resize(dims[0], dims[1]);
    
    for(int mInd=0; mInd<dims[0]*dims[1]; ++mInd){
        JBMdl mdl;
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
                
        o_mdls.GetRef(mInd) = mdl;
    }
    
//     int nMdls = mxGetNumberOfElements(i_mdls);
//     for(int mInd=0; mInd<nMdls; ++mInd){
//         
//         JBMdl mdl;
//         mdl.a = *GetDblPnt(mxGetField(i_mdls, mInd, "a"), 0, 0);
//         mdl.b = *GetDblPnt(mxGetField(i_mdls, mInd, "b"), 0, 0);
//         mdl.f = *GetIntPnt(mxGetField(i_mdls, mInd, "f"), 0, 0);
//         mdl.theta = *GetDblPnt(mxGetField(i_mdls, mInd, "theta"), 0, 0);
//         mxArray* kc = mxGetField(i_mdls, mInd, "kc");
//         mxArray* S = mxGetField(i_mdls, mInd, "S");
//         for(int cInd=0; cInd<mxGetNumberOfElements(kc); ++cInd){
//             mdl.kc.push_back(*GetDblPnt(kc, cInd, 0));
//             mdl.S.push_back(*GetIntPnt(S, cInd, 0));
//         }
//                 
//         o_mdls.push_back(mdl);
//     }
}

void ConvCMdl2MMdl(Mat<JBMdl> &i_mdls, mxArray** o_mdls){
    int nWeakLearner = i_mdls.Size(1);
    int nClsf = i_mdls.Size(2);
    if(*o_mdls == 0){
        
        int nCls = i_mdls.GetRef(0, 0).S.size();
        *o_mdls = InitMdl(nWeakLearner, nClsf, nCls);
    }
    
    int nMdls = i_mdls.Size();
    for(int mInd=0; mInd<nMdls; ++mInd){
        *GetDblPnt(mxGetField(*o_mdls, mInd, "a"), 0, 0) = i_mdls.GetRef(mInd).a;
        *GetDblPnt(mxGetField(*o_mdls, mInd, "b"), 0, 0) = i_mdls.GetRef(mInd).b;
        *GetIntPnt(mxGetField(*o_mdls, mInd, "f"), 0, 0) = i_mdls.GetRef(mInd).f;
        *GetDblPnt(mxGetField(*o_mdls, mInd, "theta"), 0, 0) = i_mdls.GetRef(mInd).theta;
        
        mxArray* kc = mxGetField(*o_mdls, mInd, "kc");
        mxArray* S = mxGetField(*o_mdls, mInd, "S");
        for(int cInd=0; cInd<i_mdls.GetRef(mInd).kc.size(); ++cInd){
            *GetDblPnt(kc, cInd, 0) = i_mdls.GetRef(mInd).kc[cInd];
            *GetIntPnt(S, cInd, 0) = i_mdls.GetRef(mInd).S[cInd];
        }
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

// struct IntImgFeat{
//     vector<double> feat;
//     int nRows;
//     int nCols;
//     int nDeps;
// };

void ConvMXIntImgFeat2CIntImgFeat(const mxArray* i_intImgFeat, vector< Mat<double> > &o_intImgFeat){
    int nImgs = mxGetNumberOfElements(i_intImgFeat);
    for(int iInd=0; iInd<nImgs; ++iInd){
//         struct IntImgFeat intImgFeat;
        mxArray *curIntImg = mxGetField(i_intImgFeat, iInd, "TextonIntImg");
        const mwSize *dims = mxGetDimensions(curIntImg);
        int nRows = dims[0];
        int nCols = dims[1];
        int nDeps = dims[2];
//         intImgFeat.feat.resize(nRows*nCols*nDeps);
        // copy
        double *mxData = (double*)mxGetData(curIntImg);
        Mat<double> intImgFeat(true, mxData, nRows, nCols, nDeps);
//         double *mxData = (double*)mxGetData(curIntImg);
//         memcpy(intImgFeat.feat.data(), mxData, nRows*nCols*nDeps*sizeof(double));
//         intImgFeat.nRows = nRows;
//         intImgFeat.nCols = nCols;
//         intImgFeat.nDeps = nDeps;
        // save
        o_intImgFeat.push_back(intImgFeat);
    }
}


// struct ixys{
//     vector<int> ixys_cls;
// };

struct SuppixInfo{
    Mat<int> label;
    Mat<int> Lbl2ID;
};
        

struct XMeta{
//     vector<struct IntImgFeat> intImgFeat;
    vector< Mat<double> > intImgFeat;
    struct TBParams TBParams;
//     vector<struct ixys> ixys;
    vector< Mat<int> > ixys;
    Mat<int> imgWHs;
    vector<struct SuppixInfo> supLabelSt;
};

void ConvMIXYS2CIXYS(const mxArray* i_ixys, vector< Mat<int> > &o_ixys){
    //init
    int nCls = mxGetNumberOfElements(i_ixys);
    o_ixys.resize(nCls);
    // set vals
    for(int cInd=0; cInd<nCls; ++cInd){
//         mxArray* ixys_cls_mx = mxGetField(i_ixys, cInd, "ixys_cls");
//         struct ixys ixys_cur;
//         int nElems = mxGetNumberOfElements(ixys_cls_mx);
//         ixys_cur.ixys_cls.resize(nElems);
//         int *mxData = (int*) mxGetData(ixys_cls_mx);
//         memcpy(ixys_cur.ixys_cls.data(), mxData, nElems*sizeof(int));   
//         // save
//         o_ixys[cInd] = ixys_cur;
        
        mxArray* ixys_cls_mx = mxGetField(i_ixys, cInd, "ixys_cls");
        int *mxData = (int*) mxGetData(ixys_cls_mx);
        int nElems = mxGetNumberOfElements(ixys_cls_mx);
        o_ixys[cInd] = Mat<int>(true, mxData, nElems, 1);
    }
}


void ConvMXMeta2CXMeta(const mxArray* i_x_meta, struct XMeta &o_x_meta){
    
    // intImgFeat
    ConvMXIntImgFeat2CIntImgFeat(mxGetField(i_x_meta, 0, "intImgFeat"), o_x_meta.intImgFeat);
    // TBParams
    ConvMTBParams2CTBParams(mxGetField(i_x_meta, 0, "TBParams"), o_x_meta.TBParams);
    // ixy
    ConvMIXYS2CIXYS(mxGetField(i_x_meta, 0, "ixy"), o_x_meta.ixys);
    // imgWHs
    mxArray *imgWHs = mxGetField(i_x_meta, 0, "imgWHs");
    int nRows = mxGetM(imgWHs);
    int nCols = mxGetN(imgWHs);
    if(nRows>0 && nCols>0){
        int* imgWHsData = (int*)mxGetData(imgWHs);
        o_x_meta.imgWHs = Mat<int>(true, imgWHsData, nRows, nCols);
//         o_x_meta.imgWHs.Resize(nRows, nCols);
//         memcpy(o_x_meta.imgWHs.Data(), imgWHsData, nRows*nCols*sizeof(int));   
    }
    // supLabelSt
    mxArray *supLabelSt = mxGetField(i_x_meta, 0, "supLabelSt");
    nRows = mxGetM(supLabelSt);
    nCols = mxGetN(supLabelSt);
    if(nRows>0 && nCols>0){
        assert(nRows == 1 || nCols == 1);
        int nCls = nRows*nCols;
        o_x_meta.supLabelSt.resize(nCls);
        for(int cInd=0; cInd<nCls; ++cInd){
            //label
            mxArray *maLabel = mxGetField(supLabelSt, cInd, "label");
            int *maLabelData = (int*)mxGetData(maLabel);
            nRows = mxGetM(maLabel);
            nCols = mxGetN(maLabel);
        
//             o_x_meta.supLabelSt[cInd].label.Resize(nRows, nCols);
//             memcpy(o_x_meta.supLabelSt[cInd].label.Data(), maLabelData, nRows*nCols*sizeof(int));   
            o_x_meta.supLabelSt[cInd].label = Mat<int>(true, maLabelData, nRows, nCols);
            //Lbl2ID
            mxArray *maLbl2ID = mxGetField(supLabelSt, cInd, "Lbl2ID");
            int *maLbl2IDData = (int*)mxGetData(maLbl2ID);
            nRows = mxGetM(maLbl2ID);
            nCols = mxGetN(maLbl2ID);
            
//             o_x_meta.supLabelSt[cInd].Lbl2ID.Resize(nRows, nCols);
//             memcpy(o_x_meta.supLabelSt[cInd].Lbl2ID.Data(), maLbl2IDData, nRows*nCols*sizeof(int));   
            o_x_meta.supLabelSt[cInd].Lbl2ID = Mat<int>(true, maLbl2IDData, nRows, nCols);
        }
    }
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

// double Geth(double x, double a, double b, double theta, double kc, int S){
double Geth(double x, int cInd, const JBMdl& mdl){
    // calc h
    double h;
    if(mdl.S[cInd] == 1){
        if(x>mdl.theta)
            h = mdl.a;
        else
            h = mdl.b;
    }else{
        h = mdl.kc[cInd];
    }
    
    // return
    return h;
}
double Geth_th(double x, double a, double b, double theta, double kc, int S){
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

// void Geths(Mat &o_hs, int i_nData, int i_nCls, vector <double> &i_xs, const mxArray *i_mdl, int i_mInd){
void Geths(Mat<double> &o_hs, int i_nData, int i_nCls, vector <double> &i_xs, JBMdl &i_mdl){
//     mxArray* a_f = mxGetField(i_mdl, i_mInd, "a");
//     mxArray* b_f = mxGetField(i_mdl, i_mInd, "b");
//     mxArray* theta_f = mxGetField(i_mdl, i_mInd, "theta");
//     mxArray* kc_f = mxGetField(i_mdl, i_mInd, "kc");
//     mxArray* S_f = mxGetField(i_mdl, i_mInd, "S");
    
    for(int cInd=0; cInd<i_nCls; ++cInd){
        for(int dInd=0; dInd<i_nData; ++dInd){
            o_hs.GetRef(dInd, cInd) = Geth(i_xs[dInd], cInd, i_mdl);
        }
    }
}

double CalcJwse(Mat<double> &i_ws, Mat<double> &i_zs, Mat<double> &i_hs){
    // o_cost = sum(sum(i_ws.*(i_zs - i_hs).^2));
    int nRows = i_ws.Size(1);
    int nCols = i_ws.Size(2);
    double ret = 0;
    for(int r=0; r<nRows; ++r)
        for(int c=0; c<nCols; ++c){
            ret += i_ws.GetRef(r, c)* //ws
                    pow(i_zs.GetRef(r, c)-i_hs.GetRef(r, c), 2.0); //(zs-hs).^2
        }
    return ret;
}

double CalcJwse_binary(Mat<double> &i_ws, Mat<double> &i_zs, Mat<double> &i_hs){
    // o_cost = sum(sum(i_ws.*(i_zs - i_hs).^2));
    int nRows = i_ws.Size(1);
    int nCols = 1;
    double ret = 0;
    for(int r=0; r<nRows; ++r)
        for(int c=0; c<nCols; ++c){
            ret += i_ws.GetRef(r, c)* //ws
                    pow(i_zs.GetRef(r, c)-i_hs.GetRef(r, c), 2.0); //(zs-hs).^2
        }
    return ret;
}


double CalcJwse_ts(Mat<double> &i_ws, Mat<double> &i_zs, vector<double> &i_xs, double a, double b, int f, double theta, vector<double> &kc, vector<int> &S){
    // o_cost = sum(sum(i_ws.*(i_zs - i_hs).^2));
    size_t nRows = i_ws.Size(1);
    size_t nCols = i_ws.Size(2);
    double ret = 0;
    for(int dInd=0; dInd<nRows; ++dInd)
        for(int cInd=0; cInd<nCols; ++cInd){
            // calc h
            double h = Geth_th(i_xs[dInd], a, b, theta, kc[cInd], S[cInd]);
            
            // i_ws.*(i_zs - i_hs).^2
            ret += i_ws.GetRef(dInd, cInd)* //ws
                    pow(i_zs.GetRef(dInd, cInd)-h, 2.0); //(zs-hs).^2
        }
    return ret;
}

double CalcJwse(Mat<double> &i_ws, Mat<double> &i_zs, vector<double> &i_xs, JBMdl& i_JBMdl){
    // o_cost = sum(sum(i_ws.*(i_zs - i_hs).^2));
    size_t nRows = i_ws.Size(1);
    size_t nCols = i_ws.Size(2);
    double ret = 0;
    for(int dInd=0; dInd<nRows; ++dInd)
        for(int cInd=0; cInd<nCols; ++cInd){
            // calc h
            double h = Geth(i_xs[dInd], cInd, i_JBMdl);
            
            // i_ws.*(i_zs - i_hs).^2
            ret += i_ws.GetRef(dInd, cInd)* //ws
                    pow(i_zs.GetRef(dInd, cInd)-h, 2.0); //(zs-hs).^2
        }
    return ret;
}

double CalcJwse_binary(Mat<double> &i_ws, Mat<double> &i_zs, vector<double> &i_xs, JBMdl& i_JBMdl){
    // o_cost = sum(sum(i_ws.*(i_zs - i_hs).^2));
    size_t nRows = i_ws.Size(1);
    int cInd = 0; // binary
    double ret = 0;
    for(int dInd=0; dInd<nRows; ++dInd){
        
        
        // calc h
        double h = Geth(i_xs[dInd], cInd, i_JBMdl);

        // i_ws.*(i_zs - i_hs).^2
        ret += i_ws.GetRef(dInd, cInd)* //ws
                pow(i_zs.GetRef(dInd, cInd)-h, 2.0); //(zs-hs).^2
    }
    return ret;
}

double GetithTextonBoost(int dInd, int fInd, struct XMeta &i_x_meta){
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
//     ixy[0] = i_x_meta.ixys[0].ixys_cls[0 + 3*dInd] - 1; // zero-base
//     ixy[1] = i_x_meta.ixys[0].ixys_cls[1 + 3*dInd] - 1; // zero-base
//     ixy[2] = i_x_meta.ixys[0].ixys_cls[2 + 3*dInd] - 1; // zero-base
    ixy[0] = i_x_meta.ixys[0][0 + 3*dInd] - 1; // zero-base
    ixy[1] = i_x_meta.ixys[0][1 + 3*dInd] - 1; // zero-base
    ixy[2] = i_x_meta.ixys[0][2 + 3*dInd] - 1; // zero-base
    
//     int nIntImgRows = i_x_meta.intImgFeat[ixy[0]].nRows;
//     int nIntImgCols = i_x_meta.intImgFeat[ixy[0]].nCols;
    int nIntImgRows = i_x_meta.intImgFeat[ixy[0]].Size(1);
    int nIntImgCols = i_x_meta.intImgFeat[ixy[0]].Size(2);
    int imgWH[2];
    imgWH[0] = nIntImgCols-1;
    imgWH[1] = nIntImgRows-1;
    
    int LOF_tl[2];
    LOF_tl[0] = ixy[1] - ((int)(LOFWH[0]-1)/2);
    LOF_tl[1] = ixy[2] - ((int)(LOFWH[1]-1)/2);
            
    int xy_part_tl[2];
    xy_part_tl[0] = LOF_tl[0] + part[0];
    xy_part_tl[1] = LOF_tl[1] + part[2];
    
    int xy_part_tl_trunk[2];
    xy_part_tl_trunk[0] = min(max(0, xy_part_tl[0]), imgWH[0]-1); // zero-base
    xy_part_tl_trunk[1] = min(max(0, xy_part_tl[1]), imgWH[1]-1); // zero-base
    
    int xy_part_br[2];
    xy_part_br[0] = LOF_tl[0] + part[1];
    xy_part_br[1] = LOF_tl[1] + part[3];
    
    int xy_part_br_trunk[2];
    xy_part_br_trunk[0] = min(max(0, xy_part_br[0]), imgWH[0]-1); // zero-base
    xy_part_br_trunk[1] = min(max(0, xy_part_br[1]), imgWH[1]-1); // zero-base
    
    // extract and return
//     double I1 = i_x_meta.intImgFeat[ixy[0]].feat[(xy_part_br_trunk[1]+1) + nIntImgRows*(xy_part_br_trunk[0]+1) + nIntImgRows*nIntImgCols*tInd];
//     double I2 = i_x_meta.intImgFeat[ixy[0]].feat[(xy_part_tl_trunk[1]) + nIntImgRows*(xy_part_br_trunk[0]+1) + nIntImgRows*nIntImgCols*tInd];
//     double I3 = i_x_meta.intImgFeat[ixy[0]].feat[(xy_part_br_trunk[1]+1) + nIntImgRows*(xy_part_tl_trunk[0]) + nIntImgRows*nIntImgCols*tInd];
//     double I4 = i_x_meta.intImgFeat[ixy[0]].feat[(xy_part_tl_trunk[1]) + nIntImgRows*(xy_part_tl_trunk[0]) + nIntImgRows*nIntImgCols*tInd];
    double I1 = i_x_meta.intImgFeat[ixy[0]][(xy_part_br_trunk[1]+1) + nIntImgRows*(xy_part_br_trunk[0]+1) + nIntImgRows*nIntImgCols*tInd];
    double I2 = i_x_meta.intImgFeat[ixy[0]][(xy_part_tl_trunk[1]) + nIntImgRows*(xy_part_br_trunk[0]+1) + nIntImgRows*nIntImgCols*tInd];
    double I3 = i_x_meta.intImgFeat[ixy[0]][(xy_part_br_trunk[1]+1) + nIntImgRows*(xy_part_tl_trunk[0]) + nIntImgRows*nIntImgCols*tInd];
    double I4 = i_x_meta.intImgFeat[ixy[0]][(xy_part_tl_trunk[1]) + nIntImgRows*(xy_part_tl_trunk[0]) + nIntImgRows*nIntImgCols*tInd];
    double Area = (xy_part_br[0] - xy_part_tl[0] + 1)*(xy_part_br[1] - xy_part_tl[1] + 1);
    return (I1-I2-I3+I4)/Area;
}


#endif