void forward(double *out, const double *y, const double *H0, int n){
  double h = 100.0;
  double V[3];
  for(int i=0;i<n;i++){
    double V[3];
    double v = 0.0;
    for(int i_=0;i_<3;i_++){
      V[i_] = H0[3*i_]*y[3*i] + H0[3*i_+1]*y[3*i+1] + H0[3*i_+2]*y[3*i+2];
    }

    for (int i_=0;i_<3;i_++){
          v += y[3*i+i_] * V[i_];
      }

    for(int i_=0;i_<3;i_++){
      for(int j_=0;j_<3;j_++){
        out[9*i+i_*3+j_] = H0[3*i_+j_] - V[i_]*V[j_]/(h+v);
      }
    }
  }  
}


void backward(double *d_y, const double *d_out, const double *y, const double *H0, int n){
  // Eigen::Vector3d yi, a;
  // Eigen::Matrix3d M, dout_dy, q, dO_dy, P;
  // for(int i=0;i<3;i++){
  //   for(int j=0;j<3;j++){
  //     M(i,j) = H0[3*i+j];
  //   }
  // }

  double h = 100.0;
  double dO_dy, P[3][3], V[3];
  for(int i=0;i<n;i++){
    double dO_dy, P[3][3], V[3];
    double v = 0.0;
    for(int i_=0;i_<3;i_++){
      V[i_] = (H0[3*i_]*y[3*i] + H0[3*i_+1]*y[3*i+1] + H0[3*i_+2]*y[3*i+2]);
    }

    for (int i_=0;i_<3;i_++){
          v += y[3*i+i_] * V[i_];
      }

    for(int i_=0;i_<3;i_++){
      V[i_] /= (h + v);
    }

    for(int i_=0;i_<3;i_++){
      for(int j_=0;j_<3;j_++){
        P[i_][j_] = V[i_]*V[j_];
      }
    }
    // out = M - M*y*y^T*M/(1 + y^T M y)
    // yi << y[3*i], y[3*i+1], y[3*i+2];
    // auto yit = yi.transpose();
    // double v = yi.dot(M*yi);
    // P = M*(yi*yit)*M;
    // for(int r=0;r<3;r++){
    //   for(int c=0;c<3;c++){
    //     q(r, c) = d_out[9*i+3*r+c];
    //   }
    // }

    for (int k=0; k<3; k++){ // compute dO/dy_k
      
      // dO_dy = - (M.col(k)* V.adjoint() + V * M.row(k))   
      //          + (2*M.row(k).dot(yi)) * P;
      //d_out  = d_out_d_y * d_y
      //d_out is a n by 9
      //d_y is a n by 3
      //d_y[3*i+k] = dO_dy.cwiseProduct(q).sum();

  

      d_y[3*i+k] = 0.0;
      //(2*M.row(k).dot(yi))
      double val = 2*(H0[3*k]*y[3*i] + H0[3*k+1]*y[3*i+1] + H0[3*k+2]*y[3*i+2]);
      for(int i_=0;i_<3;i_++){
        for(int j_=0;j_<3;j_++){
          // dO_dy = - (M.col(k)* V.adjoint() + V * M.row(k))   
          //      + (2*M.row(k).dot(yi)) * P;
          dO_dy = - (H0[3*k+i_]*V[j_] + V[i_]*H0[3*k+j_]) + P[i_][j_]*val;
          d_y[3*i+k] += dO_dy*d_out[9*i+3*i_+j_];
        }
      }
      
    }  
  }
}
