void forward(double *Fint, const double *fints, const int*el_eqns, int ngs, int neqns_per_elem){
   for(int i = 0; i<ngs; i++){
     for(int j=0;j<neqns_per_elem; j++){
        auto fint = fints + i*neqns_per_elem;
        auto el_eqns_row = el_eqns + i*neqns_per_elem;
        if(el_eqns_row[j] > 0)
          Fint[el_eqns_row[j]-1] += fint[j];
     }
   }
}

void backward(double *fint_grad, const double *Fint_grad, const double *Fint, const double *fints, const int*el_eqns, int ngs, int neqns_per_elem){
  for(int i=0;i<ngs*neqns_per_elem;i++){
    fint_grad[i] = 0.0;
  }
  for(int i = 0; i<ngs; i++){
     for(int j=0;j<neqns_per_elem; j++){
        auto el_eqns_row = el_eqns + i*neqns_per_elem;
        auto Fint_grad_ = Fint_grad + i*neqns_per_elem;
        auto fint_grad_ = fint_grad + i*neqns_per_elem;
        if(el_eqns_row[j] > 0){
          fint_grad_[j] += Fint_grad_[el_eqns_row[j]-1];
        }
     }
   }
}