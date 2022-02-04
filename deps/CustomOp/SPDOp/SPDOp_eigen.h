#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/IterativeLinearSolvers>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/LU>
#include <eigen3/Eigen/SparseQR>

void forward(double *out, const double *y, const double *H0, int n){
  Eigen::Matrix3d M;
  Eigen::Vector3d yi;
  Eigen::Matrix3d O;
  for(int i=0;i<3;i++){
    for(int j=0;j<3;j++){
      M(i,j) = H0[3*i+j];
    }
  }
  for(int i=0;i<n;i++){
    yi << y[3*i], y[3*i+1], y[3*i+2];
    auto yit = yi.transpose();
    double v = yi.dot(M*yi);
    O = M - M*(yi*yit)*M/(1.0+v);
    for(int l=0;l<3;l++){
      for(int j=0;j<3;j++){
        out[9*i+l*3+j] = O(l, j);
      }
    }
  }  
}


void backward(double *d_y, const double *d_out, const double *y, const double *H0, int n){
  Eigen::Vector3d yi, a;
  Eigen::Matrix3d M, dout_dy, q, dO_dy, P;
  for(int i=0;i<3;i++){
    for(int j=0;j<3;j++){
      M(i,j) = H0[3*i+j];
    }
  }

  for(int i=0;i<n;i++){
    // out = M - M*y*y^T*M/(1 + y^T M y)
    yi << y[3*i], y[3*i+1], y[3*i+2];
    auto yit = yi.transpose();
    double v = yi.dot(M*yi);
    P = M*(yi*yit)*M;
    for(int r=0;r<3;r++){
      for(int c=0;c<3;c++){
        q(r, c) = d_out[9*i+3*r+c];
      }
    }
    for (int k=0; k<3; k++){ // compute dO/dy_k
      a = M *yi;
      dO_dy = -1.0/(1.0 + v) * (M.col(k)* a.adjoint() + a * M.row(k))   
               + 1.0/((1.0+v)*(1.0+v))*(2*M.row(k).dot(yi)) * P;
      //d_out  = d_out_d_y * d_y
      //d_out is a n by 9
      //d_y is a n by 3
      d_y[3*i+k] = dO_dy.cwiseProduct(q).sum();
    }  
  }
}
