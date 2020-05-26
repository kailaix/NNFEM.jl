void forward_Isotropic(double *stress,const double *coef, const double*strain, int N){
  for(int i=0;i<N;i++){
    double s0 = coef[3*i], s1 = coef[3*i+1], s2 = coef[3*i+2];
    double a1 = strain[3*i], a2 = strain[3*i+1], a3 = strain[3*i+2]/2;
    stress[3*i] = s0 + s1 * a1 + s2 * (a1*a1 + a3*a3);
    stress[3*i+1] = s0 + s1 * a2 + s2 * (a2*a2+a3*a3);
    stress[3*i+2] = s1 * a3 + s2 * (a1*a3 + a2*a3);
  }
}

void forward_Isotropic(
  double *grad_coef, double *grad_strain, 
  const double * grad_stress,
  const double *stress,const double *coef, const double*strain, int N
){
  for(int i=0;i<N;i++){
    double s0 = coef[3*i], s1 = coef[3*i+1], s2 = coef[3*i+2];
    double a1 = strain[3*i], a2 = strain[3*i+1], a3 = strain[3*i+2]/2;
    grad_coef[3*i] += grad_stress[3*i] + grad_stress[3*i+1];
    grad_coef[3*i+1] = a1*grad_stress[3*i] + a2*grad_stress[3*i+1] + a3*grad_stress[3*i+2];
    grad_coef[3*i+2] = (a1*a1 + a3*a3)*grad_stress[3*i] + (a2*a2+a3*a3)*grad_stress[3*i+1] + (a1*a3 + a2*a3)*grad_stress[3*i+2];

    grad_strain[3*i] = (s1 + 2*s2*a1)*grad_stress[3*i] + (s2*a3) * grad_stress[3*i+2];
    grad_strain[3*i+1] = (s1 + 2*s2*a2)*grad_stress[3*i+1] + s2*a3*grad_stress[3*i+2];
    grad_strain[3*i+2] = 2.0*s2*a3/2.0*grad_stress[3*i] + 2.0*s2*a3/2.0*grad_stress[3*i+1] + 
                (s1 + s2*a1 + s2 *a2)/2.0*grad_stress[3*i+2];
  }
}