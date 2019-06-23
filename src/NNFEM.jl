module NNFEM
include("materials/PlaneStrain.jl")
include("solvers/Solvers.jl")
include("elements/FiniteStrainContinuum.jl")
include("fem/fem.jl")
include("fem/assembly.jl")
end 