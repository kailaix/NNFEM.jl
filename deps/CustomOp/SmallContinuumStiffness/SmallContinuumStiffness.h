#include "data.h"

int forward_count0(){
  int cnt = 0;
  for(int i=0;i<mesh.size();i++){
    for(int k=0; k<mesh[i]->nGauss;k++){
      cnt += mesh[i]->el_eqns_active.size() * mesh[i]->el_eqns_active.size();
    }
  }
  return cnt;
}

void forward_SmallContinuumStiffness(int64 *ii, int64* jj, double *vv, const double *H){
  int idx = 0.0;
  int Hidx = 0;
  for(int e=0;e<mesh.size();e++){
    auto elem = *(mesh[e]);
    for(int k=0; k<elem.nGauss;k++){
      Eigen::MatrixXd dS_dE_T = Eigen::Map<const Eigen::MatrixXd>(H+9*Hidx, 3, 3);
      Eigen::MatrixXd dS_dE = dS_dE_T.transpose();
      Hidx++;
      Eigen::MatrixXd Eu = Eigen::MatrixXd::Zero(2*elem.nnodes, 3);
      Eigen::VectorXd g1 = elem.dhdx[k].col(0), g2 = elem.dhdx[k].col(1);
      for(int i=0;i<elem.nnodes;i++){
        Eu(i,0) = g1[i];
        Eu(i,2) = g2[i];
        Eu(i+elem.nnodes, 1) = g2[i];
        Eu(i+elem.nnodes, 2) = g1[i];
      }
      Eigen::MatrixXd K = Eu * dS_dE * Eu.transpose() * elem.weights[k];

      for(int i =0; i< elem.el_eqns_active.size(); i++){
        for(int j = 0; j < elem.el_eqns_active.size(); j++){
          int ix = elem.el_eqns_active[i], iy = elem.el_eqns_active[j]; // ix, iy in (0, 2n_elem)
          int eix = elem.el_eqns[ix], eiy = elem.el_eqns[iy]; // eix, eiy in (0, 2n_domain)
          ii[idx] = eix;
          jj[idx] = eiy;
          vv[idx] = K(ix, iy);
          idx ++;
        }
      }
      
    }
  }
}



void forward_SmallContinuumStiffness(
  double *grad_H,
  const double *grad_vv, 
  const double *vv, const double *H){
  int idx = 0.0;
  int Hidx = 0;
  for(int e=0;e<mesh.size();e++){
    auto elem = *(mesh[e]);
    for(int k=0; k<elem.nGauss;k++){

      Eigen::MatrixXd local_grad_K = Eigen::MatrixXd::Zero(elem.nnodes*2, elem.nnodes*2);
      for(int i =0; i< elem.el_eqns_active.size(); i++){
        for(int j = 0; j < elem.el_eqns_active.size(); j++){
          int ix = elem.el_eqns_active[i], iy = elem.el_eqns_active[j]; // ix, iy in (0, 2n_elem)
          int eix = elem.el_eqns[ix], eiy = elem.el_eqns[iy]; // eix, eiy in (0, 2n_domain)
          // ii[idx] = eix;
          // jj[idx] = eiy;
          // vv[idx] = K(ix, iy);
          local_grad_K(ix, iy) = grad_vv[idx];
          idx ++;
        }
      }

      double * local_grad_H = grad_H+9*Hidx;
      Hidx++;
      Eigen::MatrixXd Eu = Eigen::MatrixXd::Zero(2*elem.nnodes, 3);
      Eigen::VectorXd g1 = elem.dhdx[k].col(0), g2 = elem.dhdx[k].col(1);
      for(int i=0;i<elem.nnodes;i++){
        Eu(i,0) = g1[i];
        Eu(i,2) = g2[i];
        Eu(i+elem.nnodes, 1) = g2[i];
        Eu(i+elem.nnodes, 2) = g1[i];
      }
      Eigen::MatrixXd K = Eu.transpose() * local_grad_K * Eu * elem.weights[k];
      int kk = 0;
      for(int i=0;i<3;i++){
        for(int j=0;j<3;j++){
          local_grad_H[kk++] = K(i,j);
        }
      }
      
      
    }
  }
}