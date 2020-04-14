#include "adept.h"
#include "adept_arrays.h"
#include <mutex>
#include <eigen3/Eigen/Core>
std::mutex mu;  
using namespace adept;
using Eigen::MatrixXd;
using Eigen::VectorXd;
void forward(double *stress, const double *strain, const double *strain_rate, const double *coef, int n){
  MatrixXd A(2,2), B(2,2);
  for(int i=0;i<n;i++){
    A << strain[3*i], strain[3*i+2]/2,
        strain[3*i+2]/2, strain[3*i+1];
    B << strain_rate[3*i], strain_rate[3*i+2]/2,
        strain_rate[3*i+2]/2, strain_rate[3*i+1];
    MatrixXd A2 = A * A;
    MatrixXd B2 = B * B;
    MatrixXd AB = A * B;
    MatrixXd BA = B * A;
    MatrixXd I = Eigen::MatrixXd::Identity(2,2);

    MatrixXd T = coef[0] * I + coef[1] * A + coef[2] * B + coef[3] * A2 + coef[4] * (A*B+B*A) +
        coef[5] * B2 + coef[6] * (A2*B + B * A2) + coef[7] * (A*B2+B2*A) + coef[8]*(A2*B2+B2*A2);

    stress[3*i] = T(0,0);
    stress[3*i+1] = T(1,1);
    stress[3*i+2] = T(1,0);
    coef += 9;
  }

}


void backward(
  double *grad_strain, double *grad_strain_rate, double *grad_coef, 
  const double *grad_stress, 
  const double *stress, const double *strain, const double *strain_rate, const double *coef_ipt, int n){
  const std::lock_guard<std::mutex> lock(mu);
  Stack stack;
  adouble A11, A12, A22;
  adouble B11, B12, B22;
  aVector coef(9);
  for(int i=0;i<n;i++){
    A11.set_value(strain[3*i]);
    A12.set_value(strain[3*i+2]);
    A22.set_value(strain[3*i+1]);
    B11.set_value(strain_rate[3*i]);
    B12.set_value(strain_rate[3*i+2]);
    B22.set_value(strain_rate[3*i+1]);
    for(int j=0;j<9;j++) coef[j] = coef_ipt[9*i+j];
    aMatrix22 A, B;
    Matrix22 I;
    I << 1.0, 0.0, 0.0, 1.0;
    
    stack.new_recording();
    A(0,0) = A11; A(0,1) = A12/2;
    A(0,1) = A12/2; A(1,1) = A22;
    B(0,0) = B11; B(0,1) = B12/2;
    B(0,1) = B12/2; B(1,1) = B22;
    aMatrix A2 = A * A;
    aMatrix B2 = B * B;
    aMatrix AB = A * B;
    aMatrix BA = B * A;
    
    aMatrix T = coef[0] * I + coef[1] * A + coef[2] * B + coef[3] * A2 + coef[4] * (A*B+B*A) +
        coef[5] * B2 + coef[6] * (A2*B + B * A2) + coef[7] * (A*B2+B2*A) + coef[8]*(A2*B2+B2*A2);

    adouble l = T(0,0) * grad_stress[3*i] + T(1,1) * grad_stress[3*i+1] + T(1,0) * grad_stress[3*i+2];

    l.set_gradient(1.0);
    stack.compute_adjoint();
    grad_strain[3*i] = A11.get_gradient();
    grad_strain[3*i+2] = A12.get_gradient();
    grad_strain[3*i+1] = A22.get_gradient();
    grad_strain_rate[3*i] = B11.get_gradient();
    grad_strain_rate[3*i+2] = B12.get_gradient();
    grad_strain_rate[3*i+1] = B22.get_gradient();
  }

}

