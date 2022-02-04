#include "data.h"

std::vector<Continuum*> mesh;
Domain domain;

Continuum::Continuum(const int *elnodes_, const double *coords_, 
        const double *dhdx_, const double *weights_, const double *hs_, int n_nodes, int n_gauss,
        const int *el_eqns_active_, int n_active, const int *el_eqns_){
    elnodes.resize(n_nodes);
    for (int i = 0;i<n_nodes;i++) elnodes[i] = elnodes_[i]-1;
    coords = Map<const MatrixXd>(coords_, n_nodes, 2);
    dhdx.resize(n_gauss);
    hs.resize(n_gauss);
    for(int i=0;i<n_gauss;i++){
        dhdx[i] = Map<const MatrixXd>(dhdx_, n_nodes,2);
        hs[i] = Map<const VectorXd>(hs_, n_nodes);
        hs_ += n_nodes;
        dhdx_ += 2*n_nodes;
    }
    weights = Map<const VectorXd>(weights_, n_gauss);
    

    el_eqns_active.resize(n_active);
    el_eqns.resize(2*n_nodes);
    for(int i = 0; i<n_active;i ++ ) el_eqns_active[i] = el_eqns_active_[i] - 1;
    for(int i = 0; i<2*n_nodes;i ++ ) el_eqns[i] = el_eqns_[i] - 1;

    nGauss = n_gauss;
    nnodes = n_nodes;
    
}

void create_mesh(int *elnodes, double *coords, 
        double *dhdx, double *weights, double *hs, int n_nodes, int n_gauss,
        const int *el_eqns_active_, int n_active, const int* el_eqns){
    auto c = new Continuum(elnodes, coords, dhdx, weights, hs, n_nodes, n_gauss, el_eqns_active_, n_active, el_eqns);
    mesh.push_back(c);
    domain.ngauss += n_gauss;
}

extern "C" void create_domain(const double *nodes_, int nnodes, int neqs, int neles){
    domain.nodes = Map<const MatrixXd>(nodes_, nnodes, 2);
    domain.nnodes = nnodes;
    domain.neles = neles;
    domain.neqs = neqs;
    if (neles!=mesh.size()){
        printf("Error: mesh and domain are not compatible!");
    }
}

void init_mesh(){
    domain.nodes.resize(0,0);
    domain.neqs = 0;
    domain.nnodes = 0;
    domain.neles = 0;
    domain.ngauss = 0;
    for(int i=0;i<mesh.size();i++){
        delete mesh[i];
    }
    mesh.clear();
}


void print_mesh(){
    printf("Number of Elements: %lu\nNumber of Nodes: %d\nActive DOF: %d\nNumber of Gauss points: %d\n", mesh.size(), domain.nnodes, domain.neqs, domain.ngauss);
}
