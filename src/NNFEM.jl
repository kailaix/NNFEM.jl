__precompile__(false)
module NNFEM
using PyPlot
include("utils/shapeFunctions.jl")
include("materials/PlaneStress.jl")
include("materials/PlaneStrain.jl")
include("materials/PlaneStressPlasticity.jl")
include("solvers/Solvers.jl")
include("elements/FiniteStrainContinuum.jl")
include("elements/SmallStrainContinuum.jl")
include("fem/fem.jl")
include("fem/assembly.jl")
include("utils/visualize.jl")
end 