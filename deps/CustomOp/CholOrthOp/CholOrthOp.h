void forward_CholOrthOp(double *out, const double *y, int n){
  /*     y0 y1 0
    L =     y2 0             out = L^T L
              y3
  */
  
  int ny = 4;
  for(int i=0;i<n;i++){
    out[i*9] =   y[ny*i+0] * y[ny*i+0];
    out[i*9+1] = y[ny*i+0] * y[ny*i+1];
    out[i*9+2] = 0.0;
    out[i*9+3] = y[ny*i+1] * y[ny*i+0];
    out[i*9+4] = y[ny*i+1] * y[ny*i+1] + y[ny*i+2]*y[ny*i+2];
    out[i*9+5] = 0.0;
    out[i*9+6] = 0.0;
    out[i*9+7] = 0.0;
    out[i*9+8] = y[ny*i+3] * y[ny*i+3];
  }
}


void forward_CholOrthOp(double *d_y, const double *d_out, const double *y,  int n){
  int ny = 4;
  for(int i=0;i<n;i++){
    d_y[ny*i+0] =  2*d_out[i*9]*y[ny*i+0] + (d_out[i*9+1] + d_out[i*9+3])*y[ny*i+1];
    d_y[ny*i+1] =  (d_out[i*9+1] + d_out[i*9+3])*y[ny*i+0] + 2*d_out[i*9+4]*y[ny*i+1];
    d_y[ny*i+2] =  2*d_out[i*9+4]*y[ny*i+2];
    d_y[ny*i+3] =  2*d_out[i*9+8]*y[ny*i+3];
  }
}
