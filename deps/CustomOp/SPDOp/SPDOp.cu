__global__ void forward_(double *out, const double *y, const double *H0, int n){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i>=n) return;
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
        out[9*i+i_*3+j_] = H0[3*i_+j_] - V[i_]*V[j_]/(1.0+v);
      }
    }
}

__global__ void backward_(double *d_y, const double *d_out, const double *y, const double *H0, int n){
  double dO_dy, P[3][3], V[3];
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i>=n) return;
  double v = 0.0;
  for(int i_=0;i_<3;i_++){
    V[i_] = (H0[3*i_]*y[3*i] + H0[3*i_+1]*y[3*i+1] + H0[3*i_+2]*y[3*i+2]);
  }

  for (int i_=0;i_<3;i_++){
        v += y[3*i+i_] * V[i_];
    }

  for(int i_=0;i_<3;i_++){
    V[i_] /= (1 + v);
  }

  for(int i_=0;i_<3;i_++){
    for(int j_=0;j_<3;j_++){
      P[i_][j_] = V[i_]*V[j_];
    }
  }
  for (int k=0; k<3; k++){ 
    d_y[3*i+k] = 0.0;
    double val = 2*(H0[3*k]*y[3*i] + H0[3*k+1]*y[3*i+1] + H0[3*k+2]*y[3*i+2]);
    for(int i_=0;i_<3;i_++){
      for(int j_=0;j_<3;j_++){
        dO_dy = - (H0[3*k+i_]*V[j_] + V[i_]*H0[3*k+j_]) + P[i_][j_]*val;
        d_y[3*i+k] += dO_dy*d_out[9*i+3*i_+j_];
      }
    }
    
  }  
}




void forwardGPU(double *out, const double *y, const double *H0, int n){
  forward_<<<(n+255)/256, 256>>>(out, y, H0, n);
}


void backwardGPU(double *d_y, const double *d_out, const double *y, const double *H0, int n){
  backwardGPU<<<(n+255)/256, 256>>>(d_y, d_out, y, H0, n);
}
