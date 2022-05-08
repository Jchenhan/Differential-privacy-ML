#ifndef RIDGE_H
#define RIDGE_H
#include "blackbox.hpp"
#include "regularizer.hpp"

class least_square: public blackbox {
public:
    least_square(int params_no, double* params, int regular = regularizer::ELASTIC_NET);
    least_square(double param, int regular = regularizer::L2);
    int classify(double* sample) const override;
    double zero_component_oracle_dense(double* X, double* Y, int N, double* weights = NULL) const override;
    double zero_component_oracle_sparse(double* X, double* Y, int* Jc, int* Ir, int N, double* weights = NULL) const override;
    double first_component_oracle_core_dense(double* X, double* Y, int N, int given_index, double* weights = NULL) const override;
    double first_component_oracle_core_sparse(double* X, double* Y, int* Jc, int* Ir, int N, int given_index, double* weights = NULL) const override;
};

#endif
