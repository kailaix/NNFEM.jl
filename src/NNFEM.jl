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
using ForwardDiff
using Random
using ADCMEKit

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

include("utils/options.jl")
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
include("materials/Scalar1D.jl")
include("materials/PlaneStressPlasticityLawBased.jl")
include("materials/PlaneStressIncompressibleRivlinSaunders.jl")
include("elements/FiniteStrainContinuum.jl")
include("elements/SmallStrainContinuum.jl")
include("elements/FiniteStrainTruss.jl")
include("fem/fem.jl")
include("fem/fem1D.jl")
include("fem/assembly.jl")
include("utils/Visualize.jl")
include("utils/Visualize2.jl")
include("utils/io.jl")
include("utils/Testsuit.jl")
include("utils/nnconstitutivelaw.jl")
include("utils/linearConstitutiveLaw.jl")
include("solvers/NNSolver.jl")
include("solvers/Solvers.jl")
include("solvers/SolversV2.jl")
include("solvers/AdjointSolver.jl")
include("adutils/helpers.jl")
include("adutils/kernels.jl")
include("adutils/solvers.jl")
include("adutils/representations.jl")
include("adutils/materials.jl")

end
