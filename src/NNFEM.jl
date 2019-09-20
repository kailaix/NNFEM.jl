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
sym_op = nothing
orthotropic_op = nothing
function __init__()
    global jet, cpp_fint, orthotropic_op, sym_op
    copy!(animation, pyimport("matplotlib.animation"))
    copy!(colors, pyimport("matplotlib.colors"))
    copy!(cmx, pyimport("matplotlib.cm"))
    jet = plt.get_cmap("jet")
    copy!(clb, pyimport("matplotlib.colorbar"))
    cpp_fint = load_op_and_grad("$(@__DIR__)/../deps/CustomOp/FintComp/build/libFintComp", "fint_comp")

    oplibpath = "$(@__DIR__)/../deps/CustomOp/OrthotropicOp/build/libOrthotropicOp.so"
py"""
import tensorflow as tf
lib1 = tf.load_op_library($oplibpath)
@tf.custom_gradient
def orthotropic_op(*args):
    u = lib1.orthotropic_op(*args)
    def grad(dy):
        return lib1.orthotropic_op_grad(dy, u, *args)
    return u, grad
"""
    orthotropic_op = py"orthotropic_op"


    oplibpath = "$(@__DIR__)/../deps/CustomOp/SymOp/build/libSymOp.so"
py"""
import tensorflow as tf
lib2 = tf.load_op_library($oplibpath)
@tf.custom_gradient
def sym_op(*args):
    u = lib2.sym_op(*args)
    def grad(dy):
        return lib2.sym_op_grad(dy, u, *args)
    return u, grad
"""
    sym_op = py"sym_op"
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
include("solvers/NNSolver.jl")

end