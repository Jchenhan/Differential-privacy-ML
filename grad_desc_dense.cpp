#include "grad_desc_dense.hpp"
#include "utils.hpp"
#include "regularizer.hpp"
#include <algorithm>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string.h>
#include <time.h> 
#include <stdlib.h>
#include <chrono>

extern int MAX_DIM;
const double gen = sqrt(0.50); 
const double pai=3.1415;

// Objective: 1/n * \sum_{i=1}^n{f_i(x)} + r(x), where f(x) is least_square / logistic regression,
// r(x) is a regularizer, typically L2-regularizer. 
// Ridge regression = least_square + L2
grad_desc_dense::outputs grad_desc_dense::CSGD(double* X, double* Y, int N
    , blackbox* model, int iteration_no, double step_size, double param2,int param3){
    // Random Generator
    N=28000;
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_int_distribution<int> distribution(0, N - 1);

    // Store losses and track times
    std::vector<double>* losses = new std::vector<double>;
    std::vector<double>* times = new std::vector<double>;
    std::vector<double>* test_tidu = new std::vector<double>;    
    clock_t start_ms,finish_ms;  

    int regular = model->get_regularizer();
    double* lambda = model->get_params();
    // Loss at initial guess
    N = 32561;
    losses->push_back(model->zero_oracle_dense(X, Y, N));
    times->push_back(0);
    start_ms = clock();
     N=28000;
    // Get initial guess
    double* core1 = new double[param3];
    int count =0;
    int* sample = new int[param3];
    double tidusum;
    double* x = new double[MAX_DIM];
    double* tidu = new double[MAX_DIM];
    double* avgx = new double[MAX_DIM];
    copy_vec(x, model->get_model());

    for(int i = 0; i < iteration_no; i ++) {
        // Compute a stochastic gradient: core * X[rand_samp * MAX_DIM + j]
        // is equal to "\nabla f_{rand_samp}(x)".
        tidusum=0;
        memset(tidu,0,MAX_DIM*sizeof(double));
              for(int j = 0; j < MAX_DIM; j ++){   
                for(int t = 0; t < param3; t++){
                    if(j==0){
                         sample[t] = distribution(generator);
                         core1[t] = model->first_component_oracle_core_dense(X, Y, N, sample[t], x); //constant part of gradient
                    }
                    
                    tidu[j] += core1[t] * X[sample[t] * MAX_DIM + j] / param3;  // sample[t] change to random sample  

                }
                tidusum += tidu[j]* tidu[j];
                

            }
        //test_tidu->push_back(tidusum);
         if(sqrt(tidusum) < param2){
            for(int j = 0; j < MAX_DIM; j ++) {     
                x[j] -= step_size * tidu[j] ;
                avgx[j] = (avgx[j] * i + x[j])/(i+1);
            }      
        }
        else{
            for(int j = 0; j < MAX_DIM; j ++) {                
                x[j] -= step_size * tidu[j] * param2 / sqrt(tidusum) ;
                avgx[j] = (avgx[j] * i + x[j])/(i+1);
                
            }         
        }
        // Store current loss and time every passes
        if (floor(param3 *i / N) == count) {
            losses->push_back(model->zero_oracle_dense(X, Y, N, avgx));  //seems using x is better than avgx
            finish_ms = clock();
            times->push_back((finish_ms -start_ms)/CLOCKS_PER_SEC);
            count++;
            if(count == 30) { break;}
        }
    }
    model->update_model(x);

    delete[] x;
    delete[] tidu;
    delete[] sample;
    delete[] avgx;
    delete[] core1;
    return grad_desc_dense::outputs(losses, times,test_tidu);

}

/// TODO: Clipped_dpSGD
// param2: variance param3:batchsize  param4: tidu threshold
grad_desc_dense::outputs grad_desc_dense::Clipped_dpSGD(double* X, double* Y, int N
    , blackbox* model, int iteration_no, double step_size, double param2,int param3, double param4) {
    // Random Generator
    N=28000;
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_int_distribution<int> distribution(0, N - 1);
    
    // Random Generator noise
    std::default_random_engine e;
    std::normal_distribution<double> norm(0,param2);
    
    // Store losses and track times
    std::vector<double>* losses = new std::vector<double>;
    std::vector<double>* times = new std::vector<double>;
    std::vector<double>* test_tidu = new std::vector<double>;
    clock_t start_ms,finish_ms;
   
    int regular = model->get_regularizer();
    double* lambda = model->get_params();

    // Loss at initial guess
    N=32561;
    losses->push_back(model->zero_oracle_dense(X, Y, N));
    times->push_back(0);
    //gettimeofday(&tp, NULL);
    start_ms = clock();//start_ms = tp.tv_sec * 1000 + tp.tv_usec / 1000;
    N=28000;
    // Get initial guess
    double* x = new double[MAX_DIM];
    double* Fnorm = new double[MAX_DIM];
    double* tidu = new double[MAX_DIM];
    double* avgx = new double[MAX_DIM];
    double* core1 = new double[param3];
    int count =0;
    int* sample = new int[param3];
    double tidusum;
    copy_vec(x, model->get_model());
   
    for(int i = 0; i < iteration_no; i ++) {
        /// TODO: Implement differentially private SGD
        // Compute a stochastic gradient: core * X[rand_samp * MAX_DIM + j]
        // is equal to "\nabla f_{rand_samp}(x)"
        memset(tidu,0,MAX_DIM*sizeof(double));
        tidusum=0;
            for(int j = 0; j < MAX_DIM; j ++){   
                for(int t = 0; t < param3; t++){
                    if(j==0){
                         sample[t] = distribution(generator);
                         core1[t] = model->first_component_oracle_core_dense(X, Y, N, sample[t], x); //constant part of gradient
                    }
                    
                    tidu[j] += core1[t] * X[sample[t] * MAX_DIM + j] / param3;  // sample[t] change to random sample  

                }
                tidusum += tidu[j]* tidu[j];
               

            }     
           //test_tidu->push_back(sqrt(tidusum));
        if(sqrt(tidusum) < param4){
            for(int j = 0; j < MAX_DIM; j ++) {
                double rand_samp1 = norm(e);
                x[j] -= step_size * (tidu[j] + rand_samp1 );
                avgx[j] = (avgx[j] * i + x[j])/(i+1); 
            }      
        }
        else{
            for(int j = 0; j < MAX_DIM; j ++) {
                double rand_samp1 = norm(e);                 
                x[j] -= step_size * (tidu[j] * param4 / sqrt(tidusum) +rand_samp1 );
                avgx[j] = (avgx[j] * i + x[j])/(i+1);          
            }         
        }
               
        // Store current loss and time every passes
        if (floor(param3 *i / N) == count) {
            losses->push_back(model->zero_oracle_dense(X, Y, N, avgx));  //seems using x is better than avgx
            finish_ms = clock();//gettimeofday(&tp, NULL);
            times->push_back((finish_ms -start_ms)/CLOCKS_PER_SEC);//times->push_back(tp.tv_sec * 1000 + tp.tv_usec / 1000 - start_ms);
            count++;
            if(count == 30) { 
            break;}
        }
    }
    model->update_model(x);

    delete[] x;
    delete[] tidu;
    delete[] sample;
    delete[] avgx;
    delete[] core1;
    return grad_desc_dense::outputs(losses, times,test_tidu);
}

/// TODO: DP_GD 
grad_desc_dense::outputs grad_desc_dense::DP_GD(double* X, double* Y, int N
    , blackbox* model, int iteration_no, double step_size) {
    // Random Generator;
    N=28000;
    std::default_random_engine gene;
    double epi=(sqrt(log(N)+2.0)-sqrt(log(N)))*(sqrt( log(N)+2.0)-sqrt(log(N)));
    double sigma =8*1*MAX_DIM*iteration_no/(9*log(30/0.01)*N*epi); 
    std::normal_distribution<double> norm(0,sigma);

    // Store losses and track times
    std::vector<double>* losses = new std::vector<double>;
    std::vector<double>* times = new std::vector<double>;
     std::vector<double>* test_tidu = new std::vector<double>;
    clock_t start_ms,finish_ms;  

    // Loss at initial guess
    N=32561;
    losses->push_back(model->zero_oracle_dense(X, Y, N));
    times->push_back(0);
    start_ms = clock();
    N=28000;
    // Get initial guess
    double* x = new double[MAX_DIM];
    double* tidu = new double[MAX_DIM];   
    copy_vec(x, model->get_model());
    //store original gradient at each iteration
    double a,b;
    // correction parameter
    double c,d,e,f,g,h;
    double s=3*N/(2*log(1/0.01));//variance bound set 0.5
    s=sqrt(s);
    double beta=log(1/0.01);
    test_tidu->push_back(sigma);

    for(int i = 0; i < iteration_no; i ++) {
        /// TODO: Implement differentially private GD
        // Compute a gradient: core * X[rand_samp * MAX_DIM + j]
        // is equal to "\nabla f(x)".    
       memset(tidu,0,MAX_DIM*sizeof(double));
       for(int t = 0; t < N; t++){
            double core = model->first_component_oracle_core_dense(X, Y, N, t, x);
            for(int j = 0; j < MAX_DIM; j++) {
            // Update: x = x - eta * (grad f + noise).
                a=(core * X[t * MAX_DIM + j])/s;
                b=fabs(a)/sqrt(beta)+0.01;
                c=(gen-a)/b;d=(gen+a)/b;e=0.5 * erfc(c/gen);f=0.5 * erfc(d/gen); g=exp(-c*c/2);h=exp(-d*d/2);
                tidu[j] += ((a*s*(1-(a*a)/(2*beta
                        ))-(a*a*a*s)/6 )/N+ (2*gen/3*(e-f)-(a-a*a*a/6)*(e+f)+b/(gen*sqrt(pai))*(1-a*a/2)*(h-g)+a*b*b/2*(e+f+1/(gen*sqrt(pai))*(d*h+c*g))+b*b*b/(6*gen*sqrt(pai))*((2+c*c)*g-(2+d*d)*h))/N);
                        
            }
              
        }
      test_tidu->push_back(tidu[i]);
      for(int j = 0; j < MAX_DIM; j ++) {
           double rand_samp1 = norm(gene);
           x[j] -= step_size* (tidu[j] + rand_samp1 );
          // if(i==0){
           //     test_tidu->push_back(x[j]);
          //    }
           
      }
     // Store loss and time for every passes
        
    losses->push_back(model->zero_oracle_dense(X, Y, N, x));
    finish_ms = clock();
    times->push_back((finish_ms -start_ms)/CLOCKS_PER_SEC);
    }
    model->update_model(x);

    delete[] x;
    delete[] tidu;
    return grad_desc_dense::outputs(losses, times,test_tidu);
}
























