#include "adept.h"
#include "adept_arrays.h"
#include <mutex>
#include <eigen3/Eigen/Core>
std::mutex mu0;  
using namespace adept;
using Eigen::MatrixXd;
using Eigen::VectorXd;
void forward_IsotropicTwo(double *stress, const double *strain, const double *strain_rate, const double *coef, int n){
  MatrixXd A(2,2), B(2,2), I(2,2);
  I << 1.0, 0.0, 0.0, 1.0;
  for(int i=0;i<n;i++){
    A << strain[3*i], strain[3*i+2]/2,
        strain[3*i+2]/2, strain[3*i+1];
    B << strain_rate[3*i], strain_rate[3*i+2]/2,
        strain_rate[3*i+2]/2, strain_rate[3*i+1];
    MatrixXd A2 = A * A;
    MatrixXd B2 = B * B;
    MatrixXd AB = A * B;
    MatrixXd BA = B * A;
    

    MatrixXd T = coef[0] * I + coef[1] * A + coef[2] * B + coef[3] * A2 + coef[4] * (A*B+B*A) +
        coef[5] * B2 + coef[6] * (A2*B + B * A2) + coef[7] * (A*B2+B2*A) + coef[8]*(A2*B2+B2*A2);

    stress[3*i] = T(0,0);
    stress[3*i+1] = T(1,1);
    stress[3*i+2] = T(1,0);

    coef += 9;
  }

}


void forward_IsotropicTwo(
  double *grad_strain, double *grad_strain_rate, double *grad_coef, 
  const double *grad_stress, 
  const double *stress, const double *strain, const double *strain_rate, const double *coef_ipt, int n){
  const std::lock_guard<std::mutex> lock(mu0);
  Stack stack;
  aVector coef(9);
  for(int i=0;i<9;i++) grad_coef[i] = 0.0;
  for(int i=0;i<n;i++){
    for(int j=0;j<9;j++) coef[j] = coef_ipt[j + 9*i];
    Matrix22 I;
    I << 1.0, 0.0, 0.0, 1.0;
    aMatrix22 A , B;
    A(0,0) = strain[3*i]; A(0,1) = strain[3*i+2]/2;
    A(1,0) = strain[3*i+2]/2; A(1,1) = strain[3*i+1];
    B(0,0) = strain_rate[3*i]; B(0,1) = strain_rate[3*i+2]/2;
    B(1,0) = strain_rate[3*i+2]/2; B(1,1) = strain_rate[3*i+1];
    
    stack.new_recording();
  
    aMatrix A2 = A ** A;
    aMatrix B2 = B ** B;
    aMatrix AB = A ** B;
    aMatrix BA = B ** A;
    
    aMatrix T = coef[0] * I + coef[1] * A + coef[2] * B + coef[3] * A2 + coef[4] * (A**B+B**A) +
        coef[5] * B2 + coef[6] * (A2**B + B ** A2) + coef[7] * (A**B2+B2**A) + coef[8]*(A2**B2+B2**A2);

    adouble l = T(0,0) * grad_stress[3*i] + T(1,1) * grad_stress[3*i+1] + T(1,0) * grad_stress[3*i+2];

    std::cout << T << std::endl;
    std::cout <<  stress[3*i] << "  " <<  stress[3*i+1] << "  " <<  stress[3*i+2]<< std::endl;
    l.set_gradient(1.0);
    stack.compute_adjoint();
    auto gA = A.get_gradient();
    auto gB = B.get_gradient();
    grad_strain[3*i] = gA(0,0);
    grad_strain[3*i+2] = (gA(1,0) + gA(0,1))/2;
    grad_strain[3*i+1] = gA(1,1);
    grad_strain_rate[3*i] = gB(0,0);
    grad_strain_rate[3*i+2] = (gB(0,1) + gB(1,0))/2;
    grad_strain_rate[3*i+1] = gB(1,1);
    auto gc = coef.get_gradient();
    
    for(int j=0;j<9;j++){
      grad_coef[j + 9*i] += gc(j);
    }
  }

}

