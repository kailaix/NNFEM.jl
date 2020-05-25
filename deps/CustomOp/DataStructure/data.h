#include "eigen3/Eigen/Core"
#include "vector"
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;
using Eigen::Map;

#ifdef _MSC_VER
# ifdef WIN_EXPORT
#   define EXPORTED  __declspec( dllexport )
# else
#   define EXPORTED  __declspec( dllimport )
# endif
#else
# define EXPORTED
#endif


using std::vector;
class Continuum{
public:
    VectorXi elnodes;
    MatrixXd coords;
    vector<MatrixXd> dhdx;
    Eigen::VectorXd weights;
    vector<VectorXd> hs;
    VectorXi el_eqns_active;
    VectorXi el_eqns;
    int nGauss;
    int nnodes;

    Continuum(const int *elnodes_, const double *coords_, 
        const double *dhdx_, const double *weights_, const double *hs_, int n_nodes, int n_gauss,
        const int *el_eqns_active, int n_active, const int *el_eqns);
};

class Domain{
public:
    MatrixXd nodes;
    int neqs;
    int nnodes;
    int neles;
    int ngauss;
};

extern "C" EXPORTED std::vector<Continuum*> mesh;
extern "C" EXPORTED Domain domain; 

extern "C" EXPORTED void create_mesh(int *elnodes, double *coords, 
        double *dhdx, double *weights, double *hs, int n_nodes, int n_gauss,
        const int *el_eqns_active_, int n_active, const int *el_eqns);
extern "C" EXPORTED void create_domain(const double *nodes_, int nnodes, int neqs, int neles);
extern "C" EXPORTED void init_mesh();
extern "C" EXPORTED void print_mesh();

