__precompile__(true)
module NNFEM
using PyPlot
using JLD2
using MAT
using PyCall
using Statistics
using LinearAlgebra
using PyCall:PyObject
using ADCME
using SparseArrays
animation = PyNULL()
colors = PyNULL()
cmx = PyNULL()
clb = PyNULL()

cpp_fint = nothing
function __init__()
    global jet, cpp_fint
    copy!(animation, pyimport("matplotlib.animation"))
    copy!(colors, pyimport("matplotlib.colors"))
    copy!(cmx, pyimport("matplotlib.cm"))
    jet = plt.get_cmap("jet")
    copy!(clb, pyimport("matplotlib.colorbar"))
    cpp_fint = load_op_and_grad("$(@__DIR__)/../deps/CustomOp/FintComp/build/libFintComp", "fint_comp")
end

include("utils/shapeFunctions.jl")
include("materials/PlaneStress.jl")
include("materials/PlaneStrain.jl")
include("materials/PlaneStressPlasticity.jl")
include("materials/Elasticity1D.jl")
include("materials/Plasticity1D.jl")
include("materials/Viscoplasticity1D.jl")
include("materials/NeuralNetwork2D.jl")
include("materials/NeuralNetwork1D.jl")
include("materials/PathDependent1D.jl")
include("materials/PlaneStressPlasticityLawBased.jl")
include("materials/PlaneStressIncompressibleRivlinSaunders.jl")
include("solvers/Solvers.jl")
include("elements/FiniteStrainContinuum.jl")
include("elements/SmallStrainContinuum.jl")
include("elements/FiniteStrainTruss.jl")
include("fem/fem.jl")
include("fem/assembly.jl")
include("utils/Visualize.jl")
include("utils/io.jl")
include("utils/Testsuit.jl")
include("solvers/NNSolver.jl")

end