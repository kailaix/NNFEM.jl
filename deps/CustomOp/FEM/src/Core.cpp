#ifndef __CORE_H__
#define __CORE_H__


#include "FEM.h"

//fint(d^n, S^n)
//input S[ngp*nstrain] = sigma0, sigma1 ...
//input dhdx[ngps_per_elem*neqns_per_elem]
//input weights[ngps_per_elem]
void FEM::compute_fint(){
    //loop all equations, initial to zero
    //todo only SmallStrainContinuum elem.fill_(0.0);
    //loop all Gauss
    //compute stress
    double am = -1.0;
    double beta2 = 0.5*(1 - am)*(1 - am), gamma = 0.5 - am;
    int nnodes_per_elem = neqns_per_elem/2, nstrain = 3;
    // compute nn_input, compute oeps based on od
    auto u = torch::zeros({nnodes_per_elem}, optd);
    auto v = torch::zeros({nnodes_per_elem}, optd);

    Fint.fill_(0.0);
    oeps.fill_(0.0);
    od = d + dt*v + dt*dt/2.0*((1-beta2)*a + beta2*oa);
    for(int e =0; e < nelems; e++){
        u.fill_(0.0);
        v.fill_(0.0);
        for(int i=0; i < nnodes_per_elem; i++){
            if (el_eqns_row[e*neqns_per_elem + i] > 0)
                u[i] = od[el_eqns_row[e*neqns_per_elem + i] - 1];
            if  (el_eqns_row[e*neqns_per_elem + nnodes_per_elem + i] > 0)
                v[i] = od[el_eqns_row[e*neqns_per_elem + nnodes_per_elem + i] - 1]; 
        }
            
        for(int igp = 0; igp<ngps_per_elem; igp++){
            int glo_igp = e*ngps_per_elem + igp;

            
            //g1 = self.dhdx[k][:,1]; g2 = self.dhdx[k][:,2]
            //ux = u'*g1; uy = u'*g2; vx = v'*g1; vy = v'*g2
            for(int i = 0; i < nnodes_per_elem; i++){
                oeps[glo_igp][0] += u[i]*dhdx[glo_igp*neqns_per_elem + i]; //ux
                oeps[glo_igp][1] += v[i]*dhdx[glo_igp*neqns_per_elem + nnodes_per_elem + i]; //vx
                oeps[glo_igp][2] += u[i]*dhdx[glo_igp*neqns_per_elem + nnodes_per_elem + i] + v[i]*dhdx[glo_igp*neqns_per_elem + i]; //uy+vx]
            }
        }
    }

    auto nn_input = torch::cat({oeps, eps, sigma}, 1);
    osigma = nn_predict(nn_input);

    auto fint = torch::zeros({neqns_per_elem}, optd);
    double* pE_pu_tran = new double[neqns_per_elem*nstrain];
    for(int e =0; e < nelems; e++){
        for(int igp = 0; igp<ngps_per_elem; igp++){
            int glo_igp = e*ngps_per_elem + igp;
            //comput only SmallStrainContinuum fint, associated with the Gaussin point
            fint.fill_(0.0);

            //g1 = self.dhdx[k][:,1]; g2 = self.dhdx[k][:,2]
            // compute  pE_pu_tran, neqns_per_elem by nstrain array  
            // pE_pu_tran = [g1   zeros(nnodes)    g2;
            //               zeros(nnodes)    g2   g1;] 
            for(int i=0; i < nnodes_per_elem; i++){
                pE_pu_tran[i*nstrain + 0] = dhdx[glo_igp*neqns_per_elem + i];
                pE_pu_tran[(i+nnodes_per_elem)*nstrain + 2] = dhdx[glo_igp*neqns_per_elem + i];
                pE_pu_tran[i*nstrain + 2] = dhdx[glo_igp*neqns_per_elem + nnodes_per_elem + i];
                pE_pu_tran[(i+nnodes_per_elem)*nstrain + 1] = dhdx[glo_igp*neqns_per_elem + nnodes_per_elem + i];
            }
           
        
            for(int i = 0;  i<neqns_per_elem; i++){
                for(int j =0; j< nstrain; j++) {
                    fint[i] += pE_pu_tran[i*neqns_per_elem + j]* osigma[glo_igp][j] * weights[igp]; // 1x8
                }
            }
            //loop all equations associated with the Gaussian points

            for(int j=0; j< neqns_per_elem; j++){
                if(el_eqns_row[e*neqns_per_elem + j] > 0)
                    Fint[el_eqns_row[e*neqns_per_elem + j] - 1] += fint[j];
            }
        }
    }
    ov = v + dt*((1.0-gamma)*a + gamma*oa);
    delete[] pE_pu_tran;

}

void FEM::Newton(int max_iter, double tol){
    for(int i=0;i<max_iter;i++){
        compute_residual();
        compute_jacobian();
        auto delta = -get<0>(torch::solve(J, torch::reshape(residual, {residual.size(0), 1})));
        delta = torch::reshape(delta, {delta.size(0)});
        if(torch::norm(delta).item<double>()<tol){
            return;
        }
        oa += delta;
    }
    printf("Newton does not converge in %d iterations!\n", max_iter);
}

void FEM::compute_jacobian(){
    for(int i=0;i<neqs;i++){
        residual[i].backward();
        J[i] = a.grad();
        a.grad().fill_(0.0);
    }
}

void FEM::compute_residual(){
    compute_fint();
    double am = -1.0;
    residual = torch::mm(M,((1-am)*oa + am*a)) + Fint - Fext;
}

void FEM::forward(double *poa, double *pov, double *pod, double *posigma, double *poeps){
    Newton(max_iter, tol);
    memcpy(poa, (double *)oa.data_ptr(), oa.size(0));
    memcpy(pov, (double *)ov.data_ptr(), ov.size(0));
    memcpy(pod, (double *)od.data_ptr(), od.size(0));
    memcpy(posigma, (double *)osigma.data_ptr(), osigma.size(0)*osigma.size(1));
    memcpy(poeps, (double *)oeps.data_ptr(), oeps.size(0)*oeps.size(1));
}

void FEM::backward(const double *g, double *ga, double *gv, double *gd, double *gsigma, double *geps, double* gtheta){
    int n = a.size(0);
    auto G = torch::from_blob(const_cast<double*>(g), {n,1}, optd);
    compute_residual();
    compute_jacobian();
    auto x = -torch::detach(torch::reshape(get<0>(torch::solve(J.transpose(0,1), G)), {n}));
    auto l = torch::dot(x, residual);
    l.backward();
    auto accessor_d = d.grad().accessor<double,1>();
    auto accessor_v = v.grad().accessor<double,1>();
    auto accessor_a = a.grad().accessor<double,1>();
    auto accessor_sigma = sigma.grad().accessor<double,2>();
    for(int i=0;i<d.size(0);i++) gd[i] = accessor_d[i];
    for(int i=0;i<v.size(0);i++) gv[i] = accessor_v[i];
    for(int i=0;i<a.size(0);i++) ga[i] = accessor_a[i];
    int k = 0;
    for(int i=0;i<sigma.size(0);i++){
        for(int j=0;j<sigma.size(1);j++){
            gsigma[k] = accessor_sigma[i][j];
            k++;
        }
    }
    d.grad().fill_(0.0);
    v.grad().fill_(0.0);
    a.grad().fill_(0.0);
    sigma.grad().fill_(0.0);

    k = 0;
    for(int i=0;i<nn->n_layer;i++){
        auto accessor_weight = nn->fc[i]->weight.grad().accessor<double,2>();
        auto accessor_bias = nn->fc[i]->bias.grad().accessor<double,1>();
        for(int p=0;p<nn->fc[i]->weight.size(0);p++){
            for(int q=0;q<nn->fc[i]->weight.size(1);q++){
                gtheta[k] = accessor_weight[p][q];
                k += 1;
            }
        }
        for(int p=0;p<nn->fc[i]->bias.size(0);p++){
            gtheta[k] = accessor_bias[p];
            k += 1;
        }

        nn->fc[i]->weight.grad().fill_(0.0);
        nn->fc[i]->bias.grad().fill_(0.0);
    }
    
}


void FEM::initialization(int neqs, int neqns_per_elem, int nelems, int ngps_per_elem, int ngp,
        double dt,
        const double *theta, 
        const long long *el_eqns_row,
        const double *dhdx, const double*weights,
        int max_iter, double tol,
        const double* d, const double*v, const double*a, const double*eps, const double*sigma,
        const double* Fext, const double*M,
        const double* od, const double*ov, const double*oa, const double*oeps, const double*osigma){
    FEM::neqs = neqs;
    FEM::neqns_per_elem = neqns_per_elem;
    FEM::nelems = nelems;
    FEM::ngps_per_elem = ngps_per_elem;
    FEM::ngp = ngp;
    FEM::dt = dt;
    FEM::theta = theta;
    FEM::el_eqns_row = el_eqns_row;
    FEM::dhdx = dhdx;
    FEM::weights = weights;
    FEM::max_iter = max_iter;
    FEM::tol = tol;
    FEM::d = torch::from_blob(const_cast<double*>(d), {neqs}, optd);
    FEM::v = torch::from_blob(const_cast<double*>(v), {neqs}, optd);
    FEM::a = torch::from_blob(const_cast<double*>(a), {neqs}, optd);
    FEM::eps = torch::from_blob(const_cast<double*>(eps), {neqns_per_elem*nelems, 3}, optd);
    FEM::sigma = torch::from_blob(const_cast<double*>(sigma), {neqns_per_elem*nelems, 3}, optd);
    FEM::Fext = torch::from_blob(const_cast<double*>(Fext), {neqs}, optd);
    FEM::M = torch::from_blob(const_cast<double*>(M), {neqs, neqs}, optd);
    FEM::od = torch::from_blob(const_cast<double*>(od), {neqs}, optd);
    FEM::ov = torch::from_blob(const_cast<double*>(ov), {neqs}, optd);
    FEM::oa = torch::from_blob(const_cast<double*>(oa), {neqs}, optd);
    FEM::oeps = torch::from_blob(const_cast<double*>(oeps), {neqns_per_elem*nelems, 3}, optd);
    FEM::osigma = torch::from_blob(const_cast<double*>(osigma), {neqns_per_elem*nelems, 3}, optd);
}


#endif