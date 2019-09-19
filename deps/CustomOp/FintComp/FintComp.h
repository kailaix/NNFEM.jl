void forward(double *Fint, const double *fints, const int32*el_eqns, int32 ngs, int32 neqns_per_elem, int32 neqs){
    for(int32 i=0;i<neqs;i++){
      Fint[i] = 0.0;
    }
   for(int32 i = 0; i<ngs; i++){
     for(int32 j=0;j<neqns_per_elem; j++){
        auto fint = fints + i*neqns_per_elem;
        auto el_eqns_row = el_eqns + i*neqns_per_elem;
        if(el_eqns_row[j] > 0)
          Fint[el_eqns_row[j]-1] += fint[j];
     }
   }
}

void backward(double *fint_grad, const double *Fint_grad, const double *Fint, const double *fints, const int32*el_eqns, int32 ngs, int32 neqns_per_elem, int32 neqs){
  for(int32 i=0;i<ngs*neqns_per_elem;i++){
    fint_grad[i] = 0.0;
    // printf("fint initial, %d, %f\n", i, fint_grad[i]);
  }
  // for(int32 i=0;i<neqs;i++){
  //   printf("F[%d] = %f\n", i, Fint_grad[i]);
  // }
  for(int32 i = 0; i<ngs; i++){
    auto el_eqns_row = el_eqns + i*neqns_per_elem;
    auto fint_grad_ = fint_grad + i*neqns_per_elem;
     for(int32 j=0;j<neqns_per_elem; j++){    
        if(el_eqns_row[j] > 0){
          fint_grad_[j] += Fint_grad[el_eqns_row[j]-1];
          // printf("%d/%d, %f, %f\n", el_eqns_row[j]-1, neqs, Fint_grad[el_eqns_row[j]-1], fint_grad_[j]);
        }
     }
   }
  //  for(int32 i=0; i<ngs*neqns_per_elem; i++){
  //    printf("fint, %d, %f\n", i, fint_grad[i]);
  //  }
}