# Solvers

NNFEM has two classes of solvers depending on whether differentiability is required. The first type of solvers does not support automatic differentiation, and is implemented using pure Julia. Therefore, the capability of these solvers is not restricted to the availability of differentiable kernels. Currently, the following solvers are implemented:

* [`ExplicitSolverStep`](@ref)

* [`ImplicitStaticSolver`](@ref)

* [`GeneralizedAlphaSolverStep`](@ref)

The time dependent solvers only carries out one time step advance. Therefore, in practice, users need to advance the states in time. For example,
```julia
for i = 1:NT
    global globaldata, domain = ExplicitSolverStep(globaldata, domain, Δt)
end
```

The second type of solvers supports automatic differentiation. It is implemented using `while_loop` in ADCME. Users do not need to advance in time themselves. These solvers in include:

* [`ExplicitSolver`](@ref)

* [`GeneralizedAlphaSolver`](@ref)

* [`ExplicitStaticSolver`](@ref)

* [`ImplicitStaticSolver`](@ref)

These solvers require users to prepare the external load vectors, boundary conditions, etc. To this end, NNFEM provides a set of utility functions that help compute these required quantities.

* [`ExplicitSolverTime`](@ref)

* [`compute_boundary_info`](@ref)

* [`compute_external_force`](@ref)

* [`GeneralizedAlphaSolverTime`](@ref)

