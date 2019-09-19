void forward(double *y, const double *x, int n){
  #pragma omp parallel for
  for(int i=0;i<n;i++){
    y[i*9] = x[6*i+0];
    y[i*9+1] = x[6*i+1];
    y[i*9+2] = x[6*i+2];
    y[i*9+3] = x[6*i+1];
    y[i*9+4] = x[6*i+3];
    y[i*9+5] = x[6*i+4];
    y[i*9+6] = x[6*i+2];
    y[i*9+7] = x[6*i+4];
    y[i*9+8] = x[6*i+5];
  }
}

void backward(double *grad_x, const double *grad_y, const double *y, const double *x, int n){
  for(int i=0;i<n;i++){
    grad_x[6*i+0] =  grad_y[i*9];
    grad_x[6*i+1] =  grad_y[i*9+1] + grad_y[i*9+3];
    grad_x[6*i+2] =  grad_y[i*9+2] + grad_y[i*9+6];
    grad_x[6*i+3] =  grad_y[i*9+4];
    grad_x[6*i+4] =  grad_y[i*9+5] + grad_y[i*9+7];
    grad_x[6*i+5] =  grad_y[i*9+8];
  }
}