#ifndef __NN_H__
#define __NN_H__


#include "FEM.h"
/*

Given (eps_{n+1}, eps_n, sigma_n)

       y0 y1 0
  L =     y2 0             out = L^T L
            y3

*/
void forward_chol(torch::Tensor& out, const torch::Tensor &y, const torch::Tensor &x, int n){
  
  for(int i=0;i<n;i++){
    out[i][0] = y[i][0]*y[i][0]*(x[i][0]-x[i][3]) + y[i][0]*y[i][1]*(x[i][1]-x[i][4]);
    out[i][1] = y[i][0]*y[i][1]*(x[i][0]-x[i][3]) + (y[i][1]*y[i][1]+y[i][2]*y[i][2]) * (x[i][1]-x[i][4]);
    out[i][2] = y[i][3]*y[i][3]*x[i][3];
  }
}

Net::Net(int n) {
        printf("Creating a neural network with %d layers\n", n);
        n_layer = n;
        fc.push_back( register_module("fc"+std::to_string(0), torch::nn::Linear(9, 20) ));
        for(int i=1;i<n-1;i++)
            fc.push_back( register_module("fc"+std::to_string(i), torch::nn::Linear(20, 20)) );
        fc.push_back( register_module("fc"+std::to_string(n-1), torch::nn::Linear(20, 4)) );

        for(int i=0;i<n;i++)
            fc[i]->to(torch::kDouble);
}

torch::Tensor Net::forward(torch::Tensor x) {
    torch::Tensor y = x;
    for(int i=0;i<n_layer-1;i++)
        y = torch::tanh(fc[i]->forward(y));
    y = fc[n_layer-1]->forward(y);
    int n = x.sizes()[0];
    vector<int64_t> sizes = {n, 3};
    auto out = torch::zeros(sizes, optd.requires_grad(false));
    forward_chol(out, y, x, n);
    return out;
}



// void FEM::unpack_weights(double *theta, vector<const torch::Tensor*> theta_t){
//     int start = 0;
//     for(int i=0;i<nn->n_layer;i++){
//         auto sizes = nn->fc[i]->weight.sizes();
//         *theta_t[2*i] = torch::from_blob(theta, sizes);
//         start += sizes[0]*sizes[1];
        
//         sizes = nn->fc[i]->bias.sizes();
//         *theta_t[2*i+1] = torch::from_blob(theta, sizes);
//         start += sizes[0];
//     }
// }

void FEM::nn_update_weights(vector<const torch::Tensor> theta_t){
    for(int i=0;i<nn->n_layer;i++){
        nn->fc[i]->weight.set_data(theta_t[2*i]);
        nn->fc[i]->bias.set_data(theta_t[2*i+1]);
    }     
}

void FEM::nn_grad_zero_weights(){
    for(int i=0;i<nn->n_layer;i++){
        nn->fc[i]->weight.grad().fill_(0.0);
        nn->fc[i]->bias.grad().fill_(0.0);
    }
}

torch::Tensor FEM::nn_predict(const torch::Tensor & ipt){
    auto out = nn->forward(ipt);
    return out;
}

void FEM::nn_init(){
    auto p = nn->named_parameters(false);
	auto w = p.find("weight");
	auto b = p.find("bias");

	if (w != nullptr) torch::nn::init::xavier_uniform_(*w);
	if (b != nullptr) torch::nn::init::constant_(*b, 0.01);
}


#endif