#include "data.h"
#include "adept.h"
#include "adept_arrays.h"
#include <mutex>
std::mutex mu;  
using namespace adept;


void forward(double *fint, const double *stress, const double*state){
  int idx = 0;
  for(int i=0;i<domain.neqs;i++) fint[i] = 0.0;
  int DOF = domain.nnodes;
  for(int e=0;e<mesh.size();e++){
    auto elem = *(mesh[e]);
    Eigen::VectorXd u(elem.nnodes), v(elem.nnodes);
    for(int i=0;i<elem.nnodes;i++){
      // printf("%d\n", elem->elnodes[i]);
      u[i] = state[elem.elnodes[i]];
      v[i] = state[elem.elnodes[i]+DOF];
    }

    for(int k=0; k<elem.nGauss;k++){
      Eigen::MatrixXd Eu = Eigen::MatrixXd::Zero(2*elem.nnodes, 3);
      Eigen::VectorXd g1 = elem.dhdx[k].col(0), g2 = elem.dhdx[k].col(1);
      double ux = g1.dot(u), uy = g2.dot(u), vx = g1.dot(v), vy = g2.dot(v);

      Eigen::VectorXd S(3);
      S << stress[idx], stress[idx+1], stress[idx+2]; idx+=3;
      for(int i=0;i<elem.nnodes;i++){
        Eu(i,0) = g1[i] * (1+ux);
        Eu(i,1) = g2[i] * uy;
        Eu(i,2) = g2[i] + g2[i] * ux + g1[i] * uy;
        Eu(i+elem.nnodes, 0) = g1[i] * vx;
        Eu(i+elem.nnodes, 1) = g2[i] * (1+vy);
        Eu(i+elem.nnodes, 2) = g1[i] + g1[i]*vy + g2[i]*vx;
      }
      

      Eigen::VectorXd out = Eu * S * elem.weights[k];
      // printf("****: %d\n", k);
      // std::cout << out << std::endl;
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
   double *grad_state,
  const double *grad_fint,
  const double *fint, const double *stress, const double *state){
  const std::lock_guard<std::mutex> lock(mu);
  Stack stack;
  int idx = 0;
  int DOF = domain.nnodes;
  for(int i=0;i<2*DOF;i++) grad_state[i] = 0.0;

  for(int e=0;e<mesh.size();e++){
    auto elem = *(mesh[e]);
    for(int k=0; k<elem.nGauss;k++){        
        aVector u(elem.nnodes), v(elem.nnodes), S(3);
        Vector  G1(elem.nnodes), G2(elem.nnodes);
        S << stress[idx], stress[idx+1], stress[idx+2];
        aMatrix  Eu(2*elem.nnodes,3);
        for(int i=0;i<elem.nnodes;i++){
          u(i) = state[elem.elnodes[i]];
          v(i) = state[elem.elnodes[i]+DOF];
          G1(i) = elem.dhdx[k](i,0);
          G2(i) = elem.dhdx[k](i,1);
        }
        stack.new_recording(); // important: new_recording must be here.
        auto ux = dot_product(G1, u), uy = dot_product(G2, u), 
            vx = dot_product(G1, v), vy = dot_product(G2, v);

        for(int i=0;i<elem.nnodes;i++){
          Eu(i,0) = G1(i) * (1+ux);
          Eu(i,1) = G2(i) * uy;
          Eu(i,2) = G2(i) + G2(i) * ux + G1(i) * uy;
          Eu(i+elem.nnodes, 0) = G1(i) * vx;
          Eu(i+elem.nnodes, 1) = G2(i) * (1+vy);
          Eu(i+elem.nnodes, 2) = G1(i) + G1(i)*vy + G2(i)*vx;
        }

        

        aVector afint = (Eu ** S) * elem.weights[k];
        // printf("uuuu: %d\n", k);
        // std::cout << afint << std::endl;
        adouble l = 0.0;
        // printf("size = %ld\n", elem.el_eqns_active.size());
        for(int i=0;i<elem.el_eqns_active.size();i++){
          int ix = elem.el_eqns_active[i];
          int eix = elem.el_eqns[ix];
          l += afint(ix) * grad_fint[eix];
        }
        stack.clear_gradients();
        l.set_gradient(1.0);
        stack.compute_adjoint();
        Vector grad_u = u.get_gradient(), grad_v = v.get_gradient(), grad_s = S.get_gradient();
        for(int i=0;i<elem.nnodes;i++){
          grad_state[elem.elnodes[i]] += grad_u(i);
          grad_state[elem.elnodes[i]+DOF] += grad_v(i);
        }
        // std::cout << afint.get_gradient() << std::endl;
        // std::cout << grad_u << std::endl << grad_v << std::endl << grad_s << std::endl;
        for(int i=0;i<3;i++){
            grad_stress[idx+i] = grad_s(i);
        }
        idx += 3;
    }
  }
}