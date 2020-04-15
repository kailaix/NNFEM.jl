#include "data.h"

void forward(double *strain, const double *state){
  int idx = 0;
  int DOF = domain.nnodes;
  for(int i=0;i<mesh.size();i++){
    auto elem = mesh[i];
    Eigen::VectorXd u(elem->nnodes), v(elem->nnodes);
    for(int i=0;i<elem->nnodes;i++){
      // printf("%d\n", elem->elnodes[i]);
      u[i] = state[elem->elnodes[i]];
      v[i] = state[elem->elnodes[i]+DOF];
    }
    
    for(int k=0;k<elem->nGauss;k++){
      Eigen::VectorXd g1 = elem->dhdx[k].col(0), g2 = elem->dhdx[k].col(1);
      // std::cout << g1 << std::endl;
      // std::cout << g2 << std::endl;
      // std::cout << u << std::endl;
      // std::cout << v << std::endl;
      // std::cout << "==========" << std::endl;
      strain[3*idx] = g1.dot(u);
      strain[3*idx+1] = g2.dot(v);
      strain[3*idx+2] = g1.dot(v) + g2.dot(u);
      idx++;
    }
  }
}

void backward(
   double *grad_state,
  const double *grad_strain,
  const double *strain, const double *state){
  int idx = 0;
  int DOF = domain.nnodes;
  for(int i=0;i<domain.nnodes*2;i++) grad_state[i] = 0.0;
  for(int i=0;i<mesh.size();i++){
    auto elem = mesh[i];
    
    for(int k=0;k<elem->nGauss;k++){
      Eigen::VectorXd g1 = elem->dhdx[k].col(0), g2 = elem->dhdx[k].col(1);

      for(int j=0;j<elem->nnodes;j++){
        grad_state[elem->elnodes[j]] += g1[j] * grad_strain[3*idx];
        grad_state[elem->elnodes[j]+DOF] += g2[j] * grad_strain[3*idx+1];
        grad_state[elem->elnodes[j]+DOF] += g1[j] * grad_strain[3*idx+2];
        grad_state[elem->elnodes[j]] += g2[j] * grad_strain[3*idx+2];
        
      }
      idx++;
    }
  }
}