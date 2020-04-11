# API Reference

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

## Assembly

```@docs
Domain
GlobalData
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

## Internals
```@doc

```

## Automatic Differentiation
```@doc
init_nnfem
s_eval_strain_on_gauss_points
s_compute_stiffness_matrix,
s_compute_internal_force_term
```