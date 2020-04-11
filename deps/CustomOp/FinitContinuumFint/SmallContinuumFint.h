#include "data.h"


void forward(double *fint, const double *stress){
  int idx = 0.0;
  for(int i=0;i<domain.neqs;i++) fint[i] = 0.0;
  for(int e=0;e<mesh.size();e++){
    auto elem = *(mesh[e]);
    for(int k=0; k<elem.nGauss;k++){
      Eigen::MatrixXd Eu = Eigen::MatrixXd::Zero(2*elem.nnodes, 3);
      Eigen::VectorXd g1 = elem.dhdx[k].col(0), g2 = elem.dhdx[k].col(1);
      Eigen::VectorXd S(3);
      S << stress[idx], stress[idx+1], stress[idx+2]; idx+=3;
      for(int i=0;i<elem.nnodes;i++){
        Eu(i,0) = g1[i];
        Eu(i,2) = g2[i];
        Eu(i+elem.nnodes, 1) = g2[i];
        Eu(i+elem.nnodes, 2) = g1[i];
      }
      Eigen::VectorXd out = Eu * S * elem.weights[k];
      for(int i = 0; i< elem.el_eqns_active.size(); i++){
        int ix = elem.el_eqns_active[i];
        int eix = elem.el_eqns[ix];
        fint[eix] += out[ix];
      }
    }
  }
}


void backward(
   double *grad_stress,
  const double *grad_fint,
  const double *fint, const double *stress){
  int idx = 0.0;
  for(int e=0;e<mesh.size();e++){
    auto elem = *(mesh[e]);
    for(int k=0; k<elem.nGauss;k++){
      Eigen::MatrixXd Eu = Eigen::MatrixXd::Zero(2*elem.nnodes, 3);
      Eigen::VectorXd g1 = elem.dhdx[k].col(0), g2 = elem.dhdx[k].col(1);
      for(int i=0;i<elem.nnodes;i++){
        Eu(i,0) = g1[i];
        Eu(i,2) = g2[i];
        Eu(i+elem.nnodes, 1) = g2[i];
        Eu(i+elem.nnodes, 2) = g1[i];
      }
      Eigen::VectorXd grad_out = Eigen::VectorXd::Zero(2*elem.nnodes);
      for(int i = 0; i< elem.el_eqns_active.size(); i++){
        int ix = elem.el_eqns_active[i];
        int eix = elem.el_eqns[ix];
        grad_out[ix] = grad_fint[eix];
      }
      Eigen::VectorXd g = Eu.transpose()*grad_out*elem.weights[k];
      grad_stress[idx++] = g[0];
      grad_stress[idx++] = g[1];
      grad_stress[idx++] = g[2];
      
    }
  }
}