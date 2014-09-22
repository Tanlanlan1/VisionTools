
#include "mex.h"
#include <string.h>
#include <vector>
#include <math.h>
#incluee <stdio.h>

#define DBL_MAX 2^16

using namespace std;

double* GetDblPnt(mxArray* i_m, int i_r, int i_c){
    size_t nRows = mxGetM(i_m);
    double *data = mxGetData(i_m);
    double *ret = &data[i_r + i_c*nRows];
}
int* GetIntPnt(mxArray* i_m, int i_r, int i_c){
    size_t nRows = mxGetM(i_m);
    int *data = mxGetData(i_m);
    int *ret = &data[i_r + i_c*nRows];
}

void SetDblVal(mxArray* i_m, double i_val)
{
    size_t nRows = mxGetM(i_m);
    size_t nCols = mxGetN(i_m);
    for(int r=0; r<nRows; ++r)
        for(int c=0; c<nCols; ++c)
            (*GetDblPnt(i_m, r, c)) = i_val;
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

double Jwse(mxArray* i_ws, mxArray* i_zs, mxArray* i_hs){
    // o_cost = sum(sum(i_ws.*(i_zs - i_hs).^2));
    size_t nRows = mxGetM(i_ws);
    size_t nCols = mxGetN(i_ws);
    double ret = 0;
    for(int r=0; r<nRows; ++r)
        for(int c=0; c<nCols; ++c)
            ret += (*GetDblPnt(i_ws, r, c))* //ws
                    ((*GetDblPnt(i_zs, r, c))-(*GetDblPnt(i_hs, r, c)))^2; //(zs-hs).^2
    
    return ret;
}

mxArray* InitMdl(int i_n, double i_nCls){
    mxArray *o_mdl = mxCreateStructMatrix(i_n, 1, 6, {"a", "b", "f", "theta", "kc", "S"});
    // set fields
    for(int i=0; i<i_n; ++i){
        mxSetField(o_mdl, i, "a", mxCreateNumericMatrix(1, 1, mxDOUBLE_CLASS, mxREAL));
        mxSetField(o_mdl, i, "b", mxCreateNumericMatrix(1, 1, mxDOUBLE_CLASS, mxREAL));
        mxSetField(o_mdl, i, "f", mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL));
        mxSetField(o_mdl, i, "theta", mxCreateNumericMatrix(1, 1, mxDOUBLE_CLASS, mxREAL));
        mxSetField(o_mdl, i, "kc", mxCreateNumericMatrix(i_nCls, 1, mxDOUBLE_CLASS, mxREAL));
        mxSetField(o_mdl, i, "S", mxCreateNumericMatrix(i_nCls, 1, mxDOUBLE_CLASS, mxREAL));
    }
    return o_mdl;
}

void SetMdlField(mxArray* o_mdl, double a, double b, int f, double theta, vector<double> kc, vector<double> S){
    // get
    mxArray* a_f = mxGetField(o_mdl, 0, 'a');
    mxArray* b_f = mxGetField(o_mdl, 0, 'b');
    mxArray* f_f = mxGetField(o_mdl, 0, 'f');
    mxArray* theta_f = mxGetField(o_mdl, 0, 'theta');
    mxArray* kc_f = mxGetField(o_mdl, 0, 'kc');
    mxArray* S_f = mxGetField(o_mdl, 0, 'S');
    // set
    *GetDblPnt(a_f, 0, 0) = a;
    *GetDblPnt(b_f, 0, 0) = b;
    *GetIntPnt(f_f, 0, 0) = f;
    *GetDblPnt(theta_f, 0, 0) = theta;
    for(int i=0; i<kc.size(); ++i)
        *GetDblPnt(kc_f, i, 0) = kc[i];
    for(int i=0; i<S.size(); ++i)
        *GetDblPnt(S_f, i, 0) = S[i];
}

double* GetithTextonBoost(){
    /////////////////////////////////////////////////
}


void FitStumpForAllS(
        mxArray* i_xs, mxArray* i_zs, mxArray* i_ws, mxArray* i_params, 
        mxArray* o_mdls, int o_mInd, mxArray* o_hs){
    // init
    int featDim = *((int *)mxGetData(mxGetField(i_params, 0, "featDim")));
    int nData = *((int *)mxGetData(mxGetField(i_params, 0, "nData")));
    int nCls = *((int *)mxGetData(mxGetField(i_params, 0, "nCls")));
    double featSelRatio = *((double *)mxGetData(mxGetField(i_params, 0, "featSelRatio")));
    vector<double> featValRange;
    for(int i=0; i<mxGetNumberOfElements(mxGetField(i_params, 0, "featValRange")); ++i)
        featValRange.push_back(*GetDblPnt(mxGetField(i_params, 0, "featValRange"), i, 0));
    
    mxArray* mdl_tmp = InitMdl(1, nCls);
    mxArray* hs_tmp = mxCreateDoubleMatrix(nData, nCls, mxREAL);
    
    // n* = argmin_n Jwse(n)
    double Jwse = DBL_MAX;
    mxArray* mdl_best = InitMdl(1, nCls);
    mxArray* hs_best = o_hs;
    
    // greedly select S(n) 
    vector<double> S(nCls);
    for(int maxSize=0; maxSize<nCls; ++maxSize){
        for(int candInd=0; maxSize<nCls; ++maxSize){
            if(S[candInd] == 1)
                continue;
            // choose a candidate S
            vector<double> curS = S;
            curS[candInd] = 1;
            
            // estimate k, which is independent on f, and theta 
            // fit a stump. Find a weak learner given an S
            double Jwse_S_best = DBL_MAX;
            mxArray* mdl_S_best = mdl_tmp;
            mxArray* hs_S_best = hs_tmp;
            
            for(int fInd=0; fInd<featDim; ++fInd){
                if(((double)rand()%1000)/1000 > featSelRatio)
                    continue;
                
                // wz_S = i_ws(:, curS).*i_zs(:, curS);
                // wz_nS = i_ws(:, ~curS).*i_zs(:, ~curS);
                // kc = ones(1, nCls)*nan;
                // kc(~curS) = sum(wz_nS, 1)./sum(i_ws(:, ~curS), 1); 
                
                // xs
                
                for(int tInd=0; tInd<featValRange.size(); ++tInd){
                    double curTheta = featValRange[tInd];
                    
                    // estimate a and b
                 
                    // mdl
                    
                    // calc cost 
                    
                    // keep the best
                }
                
            }
            
            // keep the best
            
        }
    }
    
     
    
    
for totSize=1:nCls
    for candInd=find(~S)
        % choose a candidate S
        curS = S;
        curS(candInd) = true;
        
        % estimate k, which is independent on f, and theta
        % fit a stump. Find a weak learner given a S
        Jwse_S_best = inf;
        mdl_S_best = mdl_init;
        hs_S_best = [];
        for fInd=1:featDim
            if rand(1) > featSelRatio
                continue;
            end
            wz_S = i_ws(:, curS).*i_zs(:, curS);
            wz_nS = i_ws(:, ~curS).*i_zs(:, ~curS);
            kc = ones(1, nCls)*nan;
            kc(~curS) = sum(wz_nS, 1)./sum(i_ws(:, ~curS), 1); 
            for tInd=1:numel(featValRange)
                curTheta = featValRange(tInd);
                
                % estimate a and b
                if isa(i_xs, 'function_handle')
                    delta_pos = i_xs(1:nData, fInd, i_x_meta) > curTheta;
                else
                    delta_pos = i_xs(:, fInd) > curTheta;
                end
                
                a = sum(sum(bsxfun(@times, wz_S, delta_pos), 1))/sum(sum(bsxfun(@times, i_ws(:, curS), delta_pos), 1));
                b = sum(sum(bsxfun(@times, wz_S, ~delta_pos), 1))/sum(sum(bsxfun(@times, i_ws(:, curS), ~delta_pos), 1));
                
                % mdl
                mdl = struct('a', a, 'b', b, 'f', fInd, 'theta', curTheta, 'kc', kc, 'S', curS);
                
                % calc cost 
                hs_S_f_t = geths(nData, nCls, delta_pos, mdl);
                Jwse_S_f_t = Jwse(i_ws, i_zs, hs_S_f_t);
                
                % keep the best
                if Jwse_S_best > Jwse_S_f_t
                    Jwse_S_best = Jwse_S_f_t;
                    mdl_S_best = mdl; 
                    hs_S_best = hs_S_f_t;
                end
            end
        end
        
        % keep the best
        if Jwse_best > Jwse_S_best
            Jwse_best = Jwse_S_best;
            mdl_best = mdl_S_best;
            S = curS;
            hs_best = hs_S_best;
        end
    end
end

%% return
o_mdl = mdl_best;
o_hs = hs_best;

    // free memory

    // return
}

mxArray* LearnJointBoost(mxArray* i_xs, mxArray* i_ys, mxArray* i_params){
    // get params
    int nWeakLearner = *((int *)mxGetData(mxGetField(i_params, 0, "nWeakLearner")));
    int nData = *((int *)mxGetData(mxGetField(i_params, 0, "nData")));
    int nCls = *((int *)mxGetData(mxGetField(i_params, 0, "nCls")));
    int verbosity = *((int *)mxGetData(mxGetField(i_params, 0, "verbosity")));
    
    // allocate zs, ws
    mxArray* zs = SetDblVal(mxCreateDoubleMatrix(nData, nCls, mxREAL), -1);
    mxArray* ws = SetDblVal(mxCreateDoubleMatrix(nData, nCls, mxREAL), 1);

    // allocate mdls
    mxArray* mdls = InitMdl(nWeakLearner, nCls);
    
    
    // init labels
    for(int dInd=0; dInd<nData; ++dInd){
        if((*GetIntPnt(i_ys, dInd, 0)) == 0) // bg
            continue;
        (*GetDblPnt(zs, (*GetIntPnt(i_ys, dInd, 1)))) = 1;
    }

    // train weak classifiers
    mxArray *hs = mxCreateDoubleMatrix(nData, nCls, mxREAL);
    for(int m=0; m<nWeakLeaner; ++m){
        if(verbosity>=1)
            printf("* boosting iter: %d/%d...", m, nWeakLearner);
        // fit a stump
        FitStumpForAllS(i_xs, zs, ws, i_params, mdls, m, hs);
        // update ws
        UpdWs(ws, zs, hs);
    
        if(verbosity >= 1)
            printf("J_wse = % 12.06f\n", Jwse(ws, zs, hs));
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
    // i_xs
    mxArray* i_xs = plhs[0];
    // double* i_ys
    mxArray* i_ys = plhs[1];
    // struct i_params
    mxArray* i_params = plhs[2];
    // learn
    mxArray* mdl = LearnJointBoost(i_xs, i_ys, i_params);
    // return
    prhs[0] = mdl;
    return;
}
