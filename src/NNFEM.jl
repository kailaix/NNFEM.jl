__precompile__(true)
module NNFEM
using PyPlot
using JLD2
using PyCall
using Statistics
using LinearAlgebra
animation = PyNULL()
colors = PyNULL()
cmx = PyNULL()
clb = PyNULL()
function __init__()
    global jet
    copy!(animation, pyimport("matplotlib.animation"))
    copy!(colors, pyimport("matplotlib.colors"))
    copy!(cmx, pyimport("matplotlib.cm"))
    jet = plt.get_cmap("jet")
    copy!(clb, pyimport("matplotlib.colorbar"))
end

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
include("utils/io.jl")
include("utils/Testsuit.jl")

end