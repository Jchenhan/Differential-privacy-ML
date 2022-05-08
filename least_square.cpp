#include "least_square.hpp"
#include <string.h>

extern int MAX_DIM;

least_square::least_square(int params_no, double* params, int regular) {
    m_params = new double[params_no];
    for(int i = 0; i < params_no; i ++)
        m_params[i] = params[i];
    m_regularizer = regular;
    m_weights = new double[MAX_DIM];
}

least_square::least_square(double param, int regular) {
    m_regularizer = regular;
    m_params = new double;
    *m_params = param;
    m_weights = new double[MAX_DIM];
}
double least_square::zero_component_oracle_dense(double* X, double* Y, int N, double* weights) const {
    if(weights == NULL) weights = m_weights;
    double _F = 0.0; N=32561;
    for(int i = 28000; i < N; i ++) {
        double _inner_xw = 0;
        for(int j = 0; j < MAX_DIM; j ++) {
            _inner_xw += weights[j] * X[i * MAX_DIM + j];
        }
        _F += (_inner_xw - Y[i]) * (_inner_xw - Y[i]);
    }
    _F /= (double)2.0 * N;
    return _F;
}

double least_square::zero_component_oracle_sparse(double* X, double* Y, int* Jc, int* Ir
    , int N, double* weights) const {
    if(weights == NULL) weights = m_weights;
    double _F = 0.0;
    for(int i = 0; i < N; i ++) {
        double _inner_xw = 0;
        for(int j = Jc[i]; j < Jc[i + 1]; j ++) {
            _inner_xw += weights[Ir[j]] * X[j];
        }
        _F += (_inner_xw - Y[i]) * (_inner_xw - Y[i]);
    }
    _F /= (double)2.0 * N;
    return _F;
}

double least_square::first_component_oracle_core_dense(double* X, double* Y, int N, int given_index, double* weights) const {
    if(weights == NULL) weights = m_weights;
    double _loss = 0, _inner_xw = 0;
    for(int j = 0; j < MAX_DIM; j ++) {
        _inner_xw += weights[j] * X[given_index * MAX_DIM + j];
    }
    _loss = _inner_xw - Y[given_index];
    return _loss;
}

double least_square::first_component_oracle_core_sparse(double* X, double* Y, int* Jc, int* Ir
    , int N, int given_index, double* weights) const {
    if(weights == NULL) weights = m_weights;
    double _loss = 0, _inner_xw = 0;
    for(int j = Jc[given_index]; j < Jc[given_index + 1]; j ++) {
        _inner_xw += weights[Ir[j]] * X[j];
    }
    _loss = _inner_xw - Y[given_index];
    return _loss;
}

int least_square::classify(double* sample) const{
    return 1;
}
