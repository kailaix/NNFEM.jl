# API Reference

## Core Data Structure
```@docs
Domain
GlobalData
```

## Core Data Structure Utilities
```@docs
getEqns
getNGauss
```

## Elements

```@autodocs
Modules = [NNFEM]
Pages   = ["FiniteStrainContinuum.jl", "SmallStrainContinuum.jl", "FiniteStrainTruss.jl"]
```


## Materials

```@autodocs
Modules = [NNFEM]
Pages   = ["PlaneStress.jl", "PlaneStrain.jl", "PlaneStressIncompressibleRivlinSaunders.jl",
            "PlaneStressPlasticity"]
```

## Matrix and Vector Assembly
```@docs
assembleInternalForce
assembleStiffAndForce
assembleMassMatrix!
getBodyForce
getExternalForce!
```

## State Updates

This set of functions include boundary condition updates, data transfer, and other bookkeeping utilities.

```@docs
commitHistory
setDirichletBoundary!
setNeumannBoundary!
updateStates!
updateDomainStateBoundary!
```

## Solvers

```@docs
ExplicitSolverStep
GeneralizedAlphaSolverStep
SolverInitial!
SolverInitial
```


## Utilities

```@autodocs
meshread
visualize_von_mises_stress
visualize_displacement
```

## Automatic Differentiation
```@docs
init_nnfem
s_eval_strain_on_gauss_points
s_compute_stiffness_matrix
s_compute_internal_force_term
f_eval_strain_on_gauss_points
f_compute_internal_force_term
ExplicitSolver
ExplicitSolverTime
GeneralizedAlphaSolver
GeneralizedAlphaSolverTime
compute_boundary_info
```