# API Reference

## Core Data Structure
```@docs
Domain
Domain(nodes::Array{Float64}, elements::Array, ndims::Int64 = 2,
    EBC::Union{missing, Array{Int64}} = missing, g::Union{missing, Array{Float64}} = missing, FBC::Union{missing, Array{Int64}} = missing, 
    f::Union{missing, Array{Float64}} = missing, edge_traction_data::Array{Int64,2}=zeros(Int64,0,3))
GlobalData
GlobalData(state::Union{Array{Float64,1},Missing},Dstate::Union{Array{Float64,1},Missing},
        velo::Union{Array{Float64,1},Missing},acce::Union{Array{Float64,1},Missing}, 
        neqs::Int64,
        EBC_func::Union{Function, Nothing}=nothing, FBC_func::Union{Function, Nothing}=nothing,
        Body_func::Union{Function,Nothing}=nothing, Edge_func::Union{Function,Nothing}=nothing)
```

### Domain 
```@docs 
getEqns
getDofs
getCoords
getNGauss
getGaussPoints
getState(domain::Domain, el_dofs::Array{Int64})
getStrain(domain::Domain)
getDStrain(domain::Domain)
getStress(domain::Domain, Δt::Float64 = 0.0; save_trace::Bool = false)
getElems
getStressHistory
getStrainHistory
getStateHistory
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
load_mesh
```

## Visualization
```@autodocs
Modules = [NNFEM]
Pages   = ["utils/Visualize2.jl"]
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