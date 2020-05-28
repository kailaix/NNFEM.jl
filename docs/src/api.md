# API Reference

## Core Data Structure
```@docs
Domain
GlobalData
```

### Domain 
```@docs 
getEqns
getDofs
getCoords
getNGauss
getGaussPoints
getState
getStress
```

## Elements

```@docs
getEdgeForce
getBodyForce
getMassMatrix
getNodes
getGaussPoints
commitHistory
```

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
getExternalForce
```

## State Updates

This set of functions include boundary condition updates, data transfer, and other bookkeeping utilities.

```@docs
commitHistory
setConstantDirichletBoundary!
setConstantNodalForces!
updateStates!
updateTimeDependentEssentialBoundaryCondition!
```

## Solvers

```@docs
ExplicitSolverStep
GeneralizedAlphaSolverStep
ImplicitStaticSolver
SolverInitial!
SolverInitial
```


## Mesh Utilities

```@docs
meshread
visualize_von_mises_stress
visualize_displacement
load_mesh
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
compute_external_force
compute_stress_rivlin_saunders
s_compute_stiffness_matrix1
```