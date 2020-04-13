void forward(double *out, const double*inp, int N){
  for(int i=0;i<N;i++){
    out[4*i] = inp[3*i];
    out[4*i+1] = inp[3*i+2]/2;
    out[4*i+2] = inp[3*i+2]/2;
    out[4*i+3] = inp[3*i+1];
  }
}

void backward(
  double *grad_inp,
  const double *grad_out,
  const double *out, const double*inp, int N){
  for(int i=0;i<N;i++){
    grad_inp[3*i] = grad_out[4*i];
    grad_inp[3*i+1] = grad_out[4*i+3];
    grad_inp[3*i+2] = grad_out[4*i+1]/2 + grad_out[4*i+2]/2;
  }
}

