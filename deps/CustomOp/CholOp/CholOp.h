void forward_CholOp(double *out, const double *y, int n){
  for(int i=0;i<n;i++){
    out[i*9] =   y[6*i+0] * y[6*i+0];
    out[i*9+1] = y[6*i+0] * y[6*i+1];
    out[i*9+2] = y[6*i+0] * y[6*i+2];
    out[i*9+3] = y[6*i+1] * y[6*i+0];
    out[i*9+4] = y[6*i+1] * y[6*i+1] + y[6*i+3]*y[6*i+3];
    out[i*9+5] = y[6*i+1] * y[6*i+2] + y[6*i+3]*y[6*i+4];
    out[i*9+6] = y[6*i+2] * y[6*i+0];
    out[i*9+7] = y[6*i+2] * y[6*i+1] + y[6*i+4]*y[6*i+3];
    out[i*9+8] = y[6*i+2] * y[6*i+2] + y[6*i+4]*y[6*i+4]+ y[6*i+5]*y[6*i+5];
  }
}


void forward_CholOp(double *d_y, const double *d_out, const double *y,  int n){
  int ny = 6;
  for(int i=0;i<n;i++){
    d_y[ny*i+0] =  2*d_out[i*9]*y[ny*i+0] + (d_out[i*9+1] + d_out[i*9+3])*y[ny*i+1] + (d_out[i*9+2] + d_out[i*9+6])*y[ny*i+2];
    d_y[ny*i+1] =  (d_out[i*9+1] + d_out[i*9+3])*y[ny*i+0] + 2*d_out[i*9+4]*y[ny*i+1] + (d_out[i*9+5] + d_out[i*9+7])*y[ny*i+2];
    d_y[ny*i+2] =  (d_out[i*9+2] + d_out[i*9+6])*y[ny*i+0] + (d_out[i*9+5] + d_out[i*9+7])*y[ny*i+1] + 2*d_out[i*9+8]*y[ny*i+2];
    d_y[ny*i+3] =  2*d_out[i*9+4]*y[ny*i+3] + (d_out[i*9+5] + d_out[i*9+7])*y[ny*i+4];
    d_y[ny*i+4] =  (d_out[i*9+5] + d_out[i*9+7])*y[ny*i+3] + 2*d_out[i*9+8]*y[ny*i+4];
    d_y[ny*i+5] =  2*d_out[i*9+8]*y[ny*i+5];
  }
}
