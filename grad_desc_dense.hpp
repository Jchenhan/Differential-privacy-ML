#ifndef GRADDESCDENSE_H
#define GRADDESCDENSE_H
#include "blackbox.hpp"

namespace grad_desc_dense {
    class outputs {
    public:
        std::vector<double>* losses;
        std::vector<double>* times;
        std::vector<double>* tidu;
        outputs(std::vector<double>* _losses, std::vector<double>* _times, std::vector<double>* _tidu):
            losses(_losses), times(_times),tidu(_tidu) {}
        outputs(std::vector<double>* _losses, std::vector<double>* _times):
            losses(_losses), times(_times) {}

    };

    outputs CSGD(double* X, double* Y, int N, blackbox* model, int iteration_no
        , double step_size, double param2,int param3);

    // TODO:
    outputs Clipped_dpSGD(double* X, double* Y, int N, blackbox* model, int iteration_no
        , double step_size, double param2, int param3, double param4);
    outputs DP_GD(double* X, double* Y, int N, blackbox* model, int iteration_no
        , double step_size);

}

#endif
