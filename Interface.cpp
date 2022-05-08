#include <iostream>
#include "mex.h"
#include "grad_desc_dense.hpp"
#include "regularizer.hpp"
#include "logistic.hpp"
#include "least_square.hpp"
#include "utils.hpp"
#include <string.h>

int MAX_DIM;
const int MAX_PARAM_STR_LEN = 15;

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    try {
        double *X = mxGetPr(prhs[0]);
        double *Y = mxGetPr(prhs[1]);
        MAX_DIM = mxGetM(prhs[0]);
        int N = mxGetN(prhs[0]);
        double *init_weight = mxGetPr(prhs[5]);
        double mu = mxGetScalar(prhs[6]);
        double L = mxGetScalar(prhs[7]);
        double param = mxGetScalar(prhs[8]);
        int iteration_no = (int) mxGetScalar(prhs[9]);
        int regularizer;
        char* _regul = new char[MAX_PARAM_STR_LEN];
        mxGetString(prhs[4], _regul, MAX_PARAM_STR_LEN);
        if(strcmp(_regul, "L2") == 0) {
            regularizer = regularizer::L2;
        }
        else if(strcmp(_regul, "L1") == 0) {
            regularizer = regularizer::L1;
        }
        else if(strcmp(_regul, "elastic_net") == 0) {
            regularizer = regularizer::ELASTIC_NET;
        }
        else mexErrMsgTxt("400 Unrecognized regularizer.");
        delete[] _regul;

        blackbox* model;
        char* _model = new char[MAX_PARAM_STR_LEN];
        mxGetString(prhs[3], _model, MAX_PARAM_STR_LEN);
        if(strcmp(_model, "logistic") == 0) {
            model = new logistic(mu, regularizer);
        }
        else if(strcmp(_model, "least_square") == 0) {
            model = new least_square(mu, regularizer);
        }
        else mexErrMsgTxt("400 Unrecognized model.");
        model->set_init_weights(init_weight);

        char* _algo = new char[MAX_PARAM_STR_LEN];
        mxGetString(prhs[2], _algo, MAX_PARAM_STR_LEN);
        double *stored_F, *stored_time; double *stored_tidu;
        int len_stored_F, len_stored_time; int len_stored_tidu;
        if(strcmp(_algo, "CSGD") == 0) {
            double clipping =  mxGetScalar(prhs[10]);
            int batch =  (int)mxGetScalar(prhs[11]);
            grad_desc_dense::outputs outputs = grad_desc_dense::CSGD(X, Y, N, model,
                iteration_no, param, clipping,batch);
            stored_F = &(*outputs.losses)[0];
            stored_time = &(*outputs.times)[0];
            stored_tidu = &(*outputs.tidu)[0];
            len_stored_F = outputs.losses->size();  
            len_stored_time = outputs.times->size();
            len_stored_tidu = outputs.tidu->size();
        }
        else if(strcmp(_algo, "Clipped_dpSGD") == 0) {
            double param2 =  mxGetScalar(prhs[10]); //noise variance
            int param3 =  (int)mxGetScalar(prhs[11]);
            double param4 =  mxGetScalar(prhs[12]);  
            grad_desc_dense::outputs outputs = grad_desc_dense::Clipped_dpSGD(X, Y, N, model,
                iteration_no, param, param2,param3, param4);
            stored_F = &(*outputs.losses)[0];
            stored_time = &(*outputs.times)[0];
            stored_tidu = &(*outputs.tidu)[0];
            len_stored_F = outputs.losses->size();
            len_stored_time = outputs.times->size();
            len_stored_tidu = outputs.tidu->size();
        }
        else if(strcmp(_algo, "DP_GD") == 0) {
            grad_desc_dense::outputs outputs = grad_desc_dense::DP_GD(X, Y, N, model,
                iteration_no, param);
            stored_F = &(*outputs.losses)[0];
            stored_time = &(*outputs.times)[0];
            stored_tidu = &(*outputs.tidu)[0];
            len_stored_F = outputs.losses->size();
            len_stored_time = outputs.times->size();
            len_stored_tidu = outputs.tidu->size();
        }
        else mexErrMsgTxt("400 Unrecognized algorithm.");
        delete[] _algo;

        plhs[0] = mxCreateDoubleMatrix(len_stored_time, 1, mxREAL);
        double* res_stored_times = mxGetPr(plhs[0]);
        for(int i = 0; i < len_stored_time; i ++)
            res_stored_times[i] = stored_time[i];
        plhs[1] = mxCreateDoubleMatrix(len_stored_F, 1, mxREAL);
        plhs[2] = mxCreateDoubleMatrix(len_stored_tidu, 1, mxREAL);
        double* res_stored_F = mxGetPr(plhs[1]);
        for(int i = 0; i < len_stored_F; i ++)
            res_stored_F[i] = stored_F[i];
        double* res_stored_tidu = mxGetPr(plhs[2]);
        for(int i = 0; i < len_stored_tidu; i ++)
            res_stored_tidu[i] = stored_tidu[i];        
        delete[] stored_F;
        delete[] stored_time;
        delete[] stored_tidu;
        delete model;
        delete[] _model;
    } catch(std::string c) {
        std::cerr << c << std::endl;
        //exit(EXIT_FAILURE);
    }
}
