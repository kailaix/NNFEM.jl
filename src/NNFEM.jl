__precompile__(true)
module NNFEM
using PyPlot
using JLD2
using MAT
using PyCall
using Statistics
using LinearAlgebra
using ADCME
using SparseArrays
animation = PyNULL()
colors = PyNULL()
cmx = PyNULL()
clb = PyNULL()

cpp_fint = nothing
sym_op = nothing
orthotropic_op = nothing
spd_op = nothing
chol_op = nothing
chol_orth_op = nothing
function __init__()
    global jet, cpp_fint, orthotropic_op, sym_op, spd_op, chol_op, chol_orth_op
    copy!(animation, pyimport("matplotlib.animation"))
    copy!(colors, pyimport("matplotlib.colors"))
    copy!(cmx, pyimport("matplotlib.cm"))
    jet = plt.get_cmap("jet")
    copy!(clb, pyimport("matplotlib.colorbar"))
    cpp_fint = load_op_and_grad("$(@__DIR__)/../deps/CustomOp/FintComp/build/libFintComp", "fint_comp")
    sym_op = load_op_and_grad("$(@__DIR__)/../deps/CustomOp/SymOp/build/libSymOp", "sym_op")
    orthotropic_op = load_op_and_grad("$(@__DIR__)/../deps/CustomOp/OrthotropicOp/build/libOrthotropicOp", "orthotropic_op")
    spd_op = load_op_and_grad("$(@__DIR__)/../deps/CustomOp/SPDOp/build/libSPDOp", "spd_op")
    chol_op = load_op_and_grad("$(@__DIR__)/../deps/CustomOp/CholOp/build/libCholOp", "chol_op")
    chol_orth_op = load_op_and_grad("$(@__DIR__)/../deps/CustomOp/CholOrthOp/build/libCholOrthOp", "chol_orth_op")
end

include("utils/shapeFunctions.jl")
include("utils/matrix.jl")
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
include("utils/nnconstitutivelaw.jl")
include("solvers/NNSolver.jl")

end