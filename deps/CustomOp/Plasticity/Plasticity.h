#include <eigen3/Eigen/Core>
#include "adept.h"
#include "adept_arrays.h"
#include <mutex>
std::mutex mu;  
using namespace adept;


using Eigen::MatrixXd;
using Eigen::VectorXd;

void forward_Plasticity(double *de, const double *h, const double *val, int N){
  MatrixXd H(3,3);
  for(int i=0;i<3;i++){
    for(int j=0;j<3;j++){
      H(i, j) = h[i+j*3];
    }
  }
  MatrixXd Ht = H.transpose();
  VectorXd g(3), f(3);
  int k = 0;
  for(int i=0;i<N;i++){
    g << val[7*i], val[7*i+1], val[7*i+2];
    f << val[7*i+3], val[7*i+4], val[7*i+5];
    double E = val[7*i + 6];
    MatrixXd DE = H - (H * g) * (Ht * f).transpose() / ( f.transpose() * H * g + E);
    for(int i_=0;i_<3;i_++){
      for(int j_=0;j_<3;j_++){
        de[k++] = DE(i_, j_);
      }
    }
    // std::cout << DE << std::endl;
  }
}

void forward_Plasticity(
  double *grad_val, double *grad_h, 
  const double *grad_de,
  const double *de, const double *h, const double *val, int N){
  const std::lock_guard<std::mutex> lock(mu);
  Stack stack;

  for(int i=0;i<9;i++) grad_h[i] = 0.0;
  
  int k = 0;
  for(int i=0;i<N;i++){
    aVector g(3), f(3);
    g << val[7*i], val[7*i+1], val[7*i+2];
    f << val[7*i+3], val[7*i+4], val[7*i+5];
    adouble E = val[7*i + 6];

    aMatrix H(3,3);
    for(int i=0;i<3;i++){
      for(int j=0;j<3;j++){
        H(i, j) = h[i+j*3];
      }
    }
    stack.new_recording();
    aMatrix Ht = H.T();
    aMatrix DE = H - (H ** g).reshape(3,1) ** (Ht ** f).reshape(1,3) / ( dot_product(f, H ** g) + E);
    adouble l = 0.0;
    for(int i_=0;i_<3;i_++){
      for(int j_=0;j_<3;j_++){
        l += DE(i_, j_) * grad_de[k++];
      }
    }
    // std::cout << DE << std::endl;
    stack.clear_gradients();
    l.set_gradient(1.0);
    stack.compute_adjoint();
    auto g_grad = g.get_gradient(), f_grad = f.get_gradient();
    double E_grad = E.get_gradient();
    for(int i_=0;i_<3;i_++) grad_val[i_] = g_grad(i_);
    for(int i_=0;i_<3;i_++) grad_val[3 + i_] = f_grad(i_);
    grad_val[6] = E_grad;
    grad_val += 7;

    auto grad_H = H.get_gradient();
    for(int i=0;i<3;i++){
      for(int j=0;j<3;j++){
        grad_h[i+j*3] += grad_H(i, j);
      }
    }
  }

  

}