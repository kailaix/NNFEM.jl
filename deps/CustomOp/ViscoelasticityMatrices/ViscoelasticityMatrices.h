#include "../DataStructure/data.h"
void forward(double *s, double *h, const double*mu_all, const double*eta_all, const double*lambda_all, double dt){
  int N = domain.ngauss;
  for(int i=0;i<N;i++){
    double mu = mu_all[i], eta = eta_all[i], lambda = lambda_all[i];
    s[9*i+0] = (0.66666666666666663*dt*mu/eta + 1)/(-0.1111111111111111*pow(dt, 2)*pow(mu, 2)/pow(eta, 2) + pow(0.66666666666666663*dt*mu/eta + 1, 2));
    s[9*i+1] = 0.33333333333333331*dt*mu/(eta*(-0.1111111111111111*pow(dt, 2)*pow(mu, 2)/pow(eta, 2) + pow(0.66666666666666663*dt*mu/eta + 1, 2)));
    s[9*i+2] = 0;
    s[9*i+3] = 0.33333333333333331*dt*mu/(eta*(-0.1111111111111111*pow(dt, 2)*pow(mu, 2)/pow(eta, 2) + pow(0.66666666666666663*dt*mu/eta + 1, 2)));
    s[9*i+4] = (0.66666666666666663*dt*mu/eta + 1)/(-0.1111111111111111*pow(dt, 2)*pow(mu, 2)/pow(eta, 2) + pow(0.66666666666666663*dt*mu/eta + 1, 2));
    s[9*i+5] = 0;
    s[9*i+6] = 0;
    s[9*i+7] = 0;
    s[9*i+8] = 1.0/(dt*mu/eta + 1);


    h[9*i+0] = 0.33333333333333331*dt*lambda*mu/(eta*(-0.1111111111111111*pow(dt, 2)*pow(mu, 2)/pow(eta, 2) + pow(0.66666666666666663*dt*mu/eta + 1, 2))) + (lambda + 2*mu)*(0.66666666666666663*dt*mu/eta + 1)/(-0.1111111111111111*pow(dt, 2)*pow(mu, 2)/pow(eta, 2) + pow(0.66666666666666663*dt*mu/eta + 1, 2));
    h[9*i+1] = 0.33333333333333331*dt*mu*(lambda + 2*mu)/(eta*(-0.1111111111111111*pow(dt, 2)*pow(mu, 2)/pow(eta, 2) + pow(0.66666666666666663*dt*mu/eta + 1, 2))) + lambda*(0.66666666666666663*dt*mu/eta + 1)/(-0.1111111111111111*pow(dt, 2)*pow(mu, 2)/pow(eta, 2) + pow(0.66666666666666663*dt*mu/eta + 1, 2));
    h[9*i+2] = 0;
    h[9*i+3] = 0.33333333333333331*dt*mu*(lambda + 2*mu)/(eta*(-0.1111111111111111*pow(dt, 2)*pow(mu, 2)/pow(eta, 2) + pow(0.66666666666666663*dt*mu/eta + 1, 2))) + lambda*(0.66666666666666663*dt*mu/eta + 1)/(-0.1111111111111111*pow(dt, 2)*pow(mu, 2)/pow(eta, 2) + pow(0.66666666666666663*dt*mu/eta + 1, 2));
    h[9*i+4] = 0.33333333333333331*dt*lambda*mu/(eta*(-0.1111111111111111*pow(dt, 2)*pow(mu, 2)/pow(eta, 2) + pow(0.66666666666666663*dt*mu/eta + 1, 2))) + (lambda + 2*mu)*(0.66666666666666663*dt*mu/eta + 1)/(-0.1111111111111111*pow(dt, 2)*pow(mu, 2)/pow(eta, 2) + pow(0.66666666666666663*dt*mu/eta + 1, 2));
    h[9*i+5] = 0;
    h[9*i+6] = 0;
    h[9*i+7] = 0;
    h[9*i+8] = mu/(dt*mu/eta + 1);
  }
}

void backward(
  double *grad_mu_all, double *grad_eta_all, double *grad_lambda_all,
  const double *grad_s, const double *grad_h,
  const double *s, const double *h, const double*mu_all, const double*eta_all, const double*lambda_all, double dt){
  int N = domain.ngauss;
  for(int i=0;i<N;i++){
    double mu = mu_all[i], eta = eta_all[i], lambda = lambda_all[i];
    double gs11 = grad_s[9*i+0];
    double gs12 = grad_s[9*i+1];
    double gs13 = grad_s[9*i+2];
    double gs21 = grad_s[9*i+3];
    double gs22 = grad_s[9*i+4];
    double gs23 = grad_s[9*i+5];
    double gs31 = grad_s[9*i+6];
    double gs32 = grad_s[9*i+7];
    double gs33 = grad_s[9*i+8];
    grad_mu_all[i] += -dt*eta*(gs33*pow(0.1111111111111111*pow(dt, 2)*pow(mu, 2) - pow(0.66666666666666663*dt*mu + eta, 2), 2) + (0.66666666666666663*dt*mu + 1.3333333333333333*eta)*pow(dt*mu + eta, 2)*(0.33333333333333331*dt*gs12*mu + 0.33333333333333331*dt*gs21*mu + gs11*(0.66666666666666663*dt*mu + eta) + gs22*(0.66666666666666663*dt*mu + eta)) + pow(dt*mu + eta, 2)*(0.1111111111111111*pow(dt, 2)*pow(mu, 2) - pow(0.66666666666666663*dt*mu + eta, 2))*(0.66666666666666663*gs11 + 0.33333333333333331*gs12 + 0.33333333333333331*gs21 + 0.66666666666666663*gs22))/(pow(dt*mu + eta, 2)*pow(0.1111111111111111*pow(dt, 2)*pow(mu, 2) - pow(0.66666666666666663*dt*mu + eta, 2), 2));
    grad_eta_all[i] += dt*mu*(gs33*pow(0.1111111111111111*pow(dt, 2)*pow(mu, 2) - pow(0.66666666666666663*dt*mu + eta, 2), 2) + (0.66666666666666663*dt*mu + 1.3333333333333333*eta)*pow(dt*mu + eta, 2)*(0.33333333333333331*dt*gs12*mu + 0.33333333333333331*dt*gs21*mu + gs11*(0.66666666666666663*dt*mu + eta) + gs22*(0.66666666666666663*dt*mu + eta)) + pow(dt*mu + eta, 2)*(0.1111111111111111*pow(dt, 2)*pow(mu, 2) - pow(0.66666666666666663*dt*mu + eta, 2))*(0.66666666666666663*gs11 + 0.33333333333333331*gs12 + 0.33333333333333331*gs21 + 0.66666666666666663*gs22))/(pow(dt*mu + eta, 2)*pow(0.1111111111111111*pow(dt, 2)*pow(mu, 2) - pow(0.66666666666666663*dt*mu + eta, 2), 2));
    grad_lambda_all[i] += 0;

    double hs11 = grad_h[9*i+0];
    double hs12 = grad_h[9*i+1];
    double hs13 = grad_h[9*i+2];
    double hs21 = grad_h[9*i+3];
    double hs22 = grad_h[9*i+4];
    double hs23 = grad_h[9*i+5];
    double hs31 = grad_h[9*i+6];
    double hs32 = grad_h[9*i+7];
    double hs33 = grad_h[9*i+8];
    grad_mu_all[i] += eta*(-dt*hs33*mu*pow(0.1111111111111111*pow(dt, 2)*pow(mu, 2) - pow(0.66666666666666663*dt*mu + eta, 2), 2) + hs33*(dt*mu + eta)*pow(0.1111111111111111*pow(dt, 2)*pow(mu, 2) - pow(0.66666666666666663*dt*mu + eta, 2), 2) - pow(dt*mu + eta, 2)*(dt*hs12*((1.0*lambda + 1.3333333333333333*mu)*(0.1111111111111111*pow(dt, 2)*pow(mu, 2) - pow(0.66666666666666663*dt*mu + eta, 2)) + (0.66666666666666663*dt*mu + 1.3333333333333333*eta)*(0.33333333333333331*dt*mu*(lambda + 2*mu) + lambda*(0.66666666666666663*dt*mu + eta))) + dt*hs21*((1.0*lambda + 1.3333333333333333*mu)*(0.1111111111111111*pow(dt, 2)*pow(mu, 2) - pow(0.66666666666666663*dt*mu + eta, 2)) + (0.66666666666666663*dt*mu + 1.3333333333333333*eta)*(0.33333333333333331*dt*mu*(lambda + 2*mu) + lambda*(0.66666666666666663*dt*mu + eta))) + hs11*(dt*(0.66666666666666663*dt*mu + 1.3333333333333333*eta)*(0.33333333333333331*dt*lambda*mu + (lambda + 2*mu)*(0.66666666666666663*dt*mu + eta)) + (0.1111111111111111*pow(dt, 2)*pow(mu, 2) - pow(0.66666666666666663*dt*mu + eta, 2))*(0.33333333333333331*dt*lambda + 1.3333333333333333*dt*mu + 0.66666666666666663*dt*(lambda + 2*mu) + 2*eta)) + hs22*(dt*(0.66666666666666663*dt*mu + 1.3333333333333333*eta)*(0.33333333333333331*dt*lambda*mu + (lambda + 2*mu)*(0.66666666666666663*dt*mu + eta)) + (0.1111111111111111*pow(dt, 2)*pow(mu, 2) - pow(0.66666666666666663*dt*mu + eta, 2))*(0.33333333333333331*dt*lambda + 1.3333333333333333*dt*mu + 0.66666666666666663*dt*(lambda + 2*mu) + 2*eta))))/(pow(dt*mu + eta, 2)*pow(0.1111111111111111*pow(dt, 2)*pow(mu, 2) - pow(0.66666666666666663*dt*mu + eta, 2), 2));
    grad_eta_all[i] += dt*mu*(hs33*mu*pow(0.1111111111111111*pow(dt, 2)*pow(mu, 2) - pow(0.66666666666666663*dt*mu + eta, 2), 2) + pow(dt*mu + eta, 2)*(hs11*((1.0*lambda + 1.3333333333333333*mu)*(0.1111111111111111*pow(dt, 2)*pow(mu, 2) - pow(0.66666666666666663*dt*mu + eta, 2)) + (0.66666666666666663*dt*mu + 1.3333333333333333*eta)*(0.33333333333333331*dt*lambda*mu + (lambda + 2*mu)*(0.66666666666666663*dt*mu + eta))) + hs12*((1.0*lambda + 0.66666666666666663*mu)*(0.1111111111111111*pow(dt, 2)*pow(mu, 2) - pow(0.66666666666666663*dt*mu + eta, 2)) + (0.66666666666666663*dt*mu + 1.3333333333333333*eta)*(0.33333333333333331*dt*mu*(lambda + 2*mu) + lambda*(0.66666666666666663*dt*mu + eta))) + hs21*((1.0*lambda + 0.66666666666666663*mu)*(0.1111111111111111*pow(dt, 2)*pow(mu, 2) - pow(0.66666666666666663*dt*mu + eta, 2)) + (0.66666666666666663*dt*mu + 1.3333333333333333*eta)*(0.33333333333333331*dt*mu*(lambda + 2*mu) + lambda*(0.66666666666666663*dt*mu + eta))) + hs22*((1.0*lambda + 1.3333333333333333*mu)*(0.1111111111111111*pow(dt, 2)*pow(mu, 2) - pow(0.66666666666666663*dt*mu + eta, 2)) + (0.66666666666666663*dt*mu + 1.3333333333333333*eta)*(0.33333333333333331*dt*lambda*mu + (lambda + 2*mu)*(0.66666666666666663*dt*mu + eta)))))/(pow(dt*mu + eta, 2)*pow(0.1111111111111111*pow(dt, 2)*pow(mu, 2) - pow(0.66666666666666663*dt*mu + eta, 2), 2));
    grad_lambda_all[i] += 1.0*eta*(hs11 + hs12 + hs21 + hs22)/(0.33333333333333331*dt*mu + 1.0*eta);
  }
}