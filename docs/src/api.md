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

```@autodocs
Modules = [NNFEM]
Pages   = ["NNSolver.jl", "Solvers.jl"]
```


## Utilities

```@autodocs
Modules = [NNFEM]
Pages   = ["io.jl", "matrix.jl", "shapeFunctions", "Testsuit.jl", "Visualize.jl", "linearConstitutiveLaw.jl"]
```

## Internals
```@doc

```