# API Reference

## Core Data Structure
```@docs
Domain
GlobalData
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
```

## State Updates

This set of functions include boundary condition updates, data transfer, and other bookkeeping utilities.

```@docs
commitHistory
setDirichletBoundary!
setNeumannBoundary!
updateStates!
updateDomainStateBoundary!
getExternalForce!
```

## Solvers

```@docs
ExplicitSolverStep
GeneralizedAlphaSolverStep
SolverInitial!
```


## Utilities

```@autodocs
Modules = [NNFEM]
Pages   = ["io.jl", "matrix.jl", "shapeFunctions", "Testsuit.jl", "Visualize.jl", "linearConstitutiveLaw.jl"]
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
```