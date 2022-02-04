#include "adept.h"
#include "adept_arrays.h"
using namespace adept;

void forward_RivlinSaunders(double * stress, const double * strain, double C1, double C2, int n){
  Matrix33 I;
  I << 1.0,0.0,0.0,
      0.0,1.0,0.0,
      0.0,0.0,1.0;
  for(int i=0;i<n;i++){
    double E11 = strain[3*i], E22 = strain[3*i+1], G12 = strain[3*i+2];
    double C11 = 2*E11+1.0, C22 = 2*E22+1.0, C12 = G12;
    double det22C = C11*C22 - C12*C12;
    double C33 = 1/det22C;
    double I1 = C11 + C22 + C33;

    Matrix33 C, Cinv;
    C << C11, C12, 0.0,
        C12, C22, 0.0,
        0.0, 0.0, C33;
    Cinv << C22/det22C, -C12/det22C, 0.0,
          -C12/det22C, C11/det22C, 0.0,
          0.0, 0.0, 1.0/C33;
    Matrix33 svol = Cinv;
    Matrix33 Siso = 2.0*C1*I + 2.0*C2*(C + I1*I);
    double p = -C33*Siso(2,2);
    stress[3*i] = value(Siso(0,0) + p*svol(0,0));
    stress[3*i+1] = value(Siso(1,1) + p*svol(1,1));
    stress[3*i+2] = value(Siso(0,1) + p*svol(0,1));
  }
}

void forward_RivlinSaunders(
  double *grad_strain, double *grad_C1, double *grad_C2,
  const double *grad_stress,
  const double * stress, const double * strain, double C1, double C2, int n){
  Stack stack;
  Matrix33 I;
  I << 1.0,0.0,0.0,
      0.0,1.0,0.0,
      0.0,0.0,1.0;
  grad_C1[0] = 0.0;
  grad_C2[0] = 0.0;
  for(int i=0;i<n;i++){
    adouble E11 = strain[3*i], E22 = strain[3*i+1], G12 = strain[3*i+2];
    adouble C1_ = C1, C2_ = C2;
    aMatrix33 C, Cinv;
    stack.new_recording();
    adouble C11 = 2*E11+1.0, C22 = 2*E22+1.0, C12 = G12;
    adouble det22C = C11*C22 - C12*C12;
    adouble C33 = 1/det22C;
    adouble I1 = C11 + C22 + C33;

    C(0,0) = C11; C(0,1) = C12; C(0,2) = 0.0;
    C(1,0) = C12; C(1,1) = C22; C(1,2) = 0.0;
    C(2,0) = 0.0; C(2,1) = 0.0; C(2,2) = C33;

    Cinv(0,0) = C22/det22C; Cinv(0,1) = -C12/det22C; Cinv(0,2) = 0.0;
    Cinv(1,0) = -C12/det22C; Cinv(1,1) = C11/det22C; Cinv(1,2) = 0.0;
    Cinv(2,0) = 0.0; Cinv(2,1) = 0.0; Cinv(2,2) = 1.0/C33;
    
    aMatrix33 svol = Cinv;
    aMatrix33 Siso = 2.0*C1_*I + 2.0*C2_*(C + I1*I);
    adouble p = -C33*Siso(2,2);

    adouble l = grad_stress[3*i] * (Siso(0,0) + p*svol(0,0)) + 
                grad_stress[3*i+1] * (Siso(1,1) + p*svol(1,1)) + 
                grad_stress[3*i+2] * (Siso(0,1) + p*svol(0,1));
   
    l.set_gradient(1.0);
    stack.compute_adjoint();
    grad_C1[0] += C1_.get_gradient();
    grad_C2[0] += C2_.get_gradient();
    grad_strain[3*i] = E11.get_gradient();
    grad_strain[3*i+1] = E22.get_gradient();
    grad_strain[3*i+2] = G12.get_gradient();
  }
}