# Automatic Differentiation

## Data Structure

To facilitate implementing custom operators, we made a shared library for storing all FEM data structures that do not participate in automatic differentiation. In the shared library, there are mainly two data structures

* `domain`

  ```c++
  class Domain{
  public:
      MatrixXd nodes;
      int neqs;
      int nnodes;
      int neles;
      int ngauss;
  };
  ```

  Here 

  * `nodes`: coordinates of all nodes, a $n_v\times 2$ matrix.
  * `neqs`: number of all active DOFs among $2n_v$ equations.
  * `nnodes`: number of nodes, $n_v$
  * `neles`: number of elements $n_e$, which is also the size of `mesh` vector (see below). 
  * `ngauss`: total number of Gauss points. It is equal to `getNGauss(domain)` in Julia. 

* `continuum`

  ```c++
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
  ```

  * `elnodes`: the global index of the nodes for this specific element, $n^e_v$.

  * `coords`: coordinates of the element vertices, it is of size $n^e_v\times2$

  * `dhdx`: a list (length = $n_g$) of  $n_v^e\times 2$ matrices, representing the contribution of  $\nabla \phi_i(x)$ to each nodes. $n_g$ is the number of Gauss points.

  * `weights`: weight vector of Gauss quadrature

  * `hs`: a list (length = $n_g$) of length $n_v^e $ vector, representing the contribution of  $\phi_i(x)$ to each nodes. $n_g$ is the number of Gauss points.

  * `el_eqns`: global indices of active DOFs for each vertex and each direction ($u$ and $v$). It has length $2n_v^e$ and each value is within ${0,1,\ldots, 2n_v-1}$. 

  * `el_eqns_active`: local indices of actives DOFs for each vertex and each direction ($u$ and $v$). It has length **at most** $2n_v^e$ and each value is within ${0,1,\ldots, 2n_v^e-1}$. A typical  usuage is 

    ```c++
    // fint: local internal force
    // Fint: global internal force
    for(int i = 0; i< elem.el_eqns_active.size(); i++){
      int ix = elem.el_eqns_active[i];
      int eix = elem.el_eqns[ix];
      Fint[eix] += fint[ix];
    }
    ```

## Examples



```julia
using NNFEM


domain = example_domain()
globaldata = example_global_data(domain)
init_nnfem(domain)  # IMPORTANT: initialize the NNFEM session

# total number of gauss points
ngauss = length(domain.elements[1].weights) * domain.neles
H = constant(rand(ngauss, 3, 3)) # linear elasticity matrix 
K = s_compute_stiffness_matrix(H, domain) # stiffness matrix
assembleMassMatrix!(globaldata, domain)
M = SparseTensor(globaldata.M) # mass matrix

a = 0.1
b = 0.2
A = a * K + b * M

rhs = constant(rand(size(K,2)))
sol = A\rhs
```

1. `sol` can be differentiated with respect to `rhs`

```
julia> gradients(sum(sol), rhs)
PyObject <tf.Tensor 'gradients_1/IdentityN_12_grad/SparseSolverGrad:3' shape=(242,) dtype=float64>
```

2. `sol` can be differentiated with respect to `H`

```
julia> gradients(sum(sol), H)
PyObject <tf.Tensor 'gradients_2/IdentityN_11_grad/SmallContinuumStiffnessGrad:0' shape=(400, 3, 3) dtype=float64>
```

---



Given the stress `stress`, we can compute the internal force and evaluate its gradients

```julia
stress = constant(rand(ngauss, 3))
fint = s_compute_internal_force_term(stress, domain)
gradients(sum(fint), stress)
```

Expected

```
PyObject <tf.Tensor 'gradients_3/IdentityN_13_grad/SmallContinuumFintGrad:0' shape=(400, 3) dtype=float64>
```

---

Given the displacement, we can evaluate the strain and evaluate the gradients

```julia
state = constant(rand(domain.nnodes*2))
strain = s_eval_strain_on_gauss_points(state, domain)
gradients(sum(strain), state)
```

Expected

```
PyObject <tf.Tensor 'gradients_4/IdentityN_14_grad/SmallContinuumStrainGrad:0' shape=(242,) dtype=float64>
```



