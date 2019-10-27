#ifndef __FEN_H__
#define __FEN_H__

#include "la.h"
#include <vector>
#include <string>

using namespace std;

struct Net : torch::nn::Module {
    Net(int n);
    torch::Tensor forward(torch::Tensor x);
    vector<torch::nn::Linear> fc;
    int n_layer;
};


class FEM
{
public:
    /* data */
    int neqs, neqns_per_elem, nelems, ngps_per_elem, ngp;
    double dt;
    int *el_eqns_row;
    double *dhdx, *weights;
    int max_iter; double tol;
    torch::Tensor d, v, a;
    torch::Tensor S, fext, M;
    torch::Tensor od, oa, osigma, oeps, eps, sigma;
    
    double *theta; 
    torch::Tensor gtheta, gw ;
    torch::Tensor residual, J, Fext, Fint;
    shared_ptr<Net> nn;

    /* neural network */
    void nn_unpack_weights(double *theta, vector<const torch::Tensor> theta_t);
    void nn_update_weights(vector<const torch::Tensor> theta_t);
    void nn_grad_zero_weights();
    torch::Tensor nn_predict(const torch::Tensor & ipt);
    void nn_init();
public:
    FEM(int n_layer){
        nn = std::make_shared<Net>(n_layer);
    };
    ~FEM(){};
    
    void compute_residual();
    void compute_gradient_theta();
    void compute_gradient_w();
    void compute_jacobian();
    void compute_fint();

    void forward();
    void backward(double *g, double *gsigma, double *gd, double *gv, double *ga, double* theta_t);

    void Newton(int max_iter, double tol);
private:
    void disp_to_strain();
};




#endif



