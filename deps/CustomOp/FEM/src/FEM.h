#ifndef __FEN_H__
#define __FEN_H__

#include <functional>
#include <cmath>
#include <iostream>
#include <torch/torch.h>
#include <vector>
#include <string>

static auto optd = torch::TensorOptions().dtype(torch::kDouble).layout(torch::kStrided).requires_grad(false);
static auto optf = torch::TensorOptions().dtype(torch::kFloat).layout(torch::kStrided).requires_grad(false);

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
    const double *theta; 
    const long long *el_eqns_row;
    const double *dhdx, *weights;
    int max_iter; double tol;
    torch::Tensor d, v, a, eps, sigma;
    torch::Tensor Fext, M;
    torch::Tensor od, ov, oa, osigma, oeps;
    
    
    torch::Tensor residual, J, Fint;
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

    FEM(): FEM(3){};
    ~FEM(){};

    void initialization(int neqs, int neqns_per_elem, int nelems, int ngps_per_elem, int ngp,
        double dt,
        const double *theta, 
        const long long *el_eqns_row,
        const double *dhdx, const double*weights,
        int max_iter, double tol,
        const double* d, const double*v, const double*a, const double*eps, const double*sigma,
        const double* Fext, const double*M,
        const double* od, const double*ov, const double*oa, const double*oeps, const double*osigma);

    void compute_residual();
    void compute_gradient_theta();
    void compute_gradient_w();
    void compute_jacobian();
    void compute_fint();

    void forward(double *oa, double *ov, double *od, double *osigma, double *oeps);
    void backward(const double *g, double *ga, double *gv, double *gd, double *gsigma, double *geps, double* gtheta);

    void Newton(int max_iter, double tol);
private:
    void disp_to_strain();
};




#endif



