# Automatic Differentiation 

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



