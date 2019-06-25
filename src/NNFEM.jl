__precompile__(false)
module NNFEM
include("utils/shapeFunctions.jl")
include("materials/PlaneStrain.jl")
include("materials/Plasticity.jl")
include("solvers/Solvers.jl")
include("elements/FiniteStrainContinuum.jl")
include("elements/SmallStrainContinuum.jl")
include("fem/fem.jl")
include("fem/assembly.jl")
include("utils/visualize.jl")
end 