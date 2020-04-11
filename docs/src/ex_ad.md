# Automatic Differentiation 

```julia
using NNFEM

domain = example_domain()
init_nnfem(domain)  # IMPORTANT: initialize the NNFEM session

K = s_compute_fem_stiffness_matrix()
```

