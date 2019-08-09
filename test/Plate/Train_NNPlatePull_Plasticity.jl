using Revise
using Test 
using NNFEM
using PyCall
using PyPlot
using JLD2
using ADCME
using LinearAlgebra
reset_default_graph()
include("nnutil.jl")

testtype = "NeuralNetwork2D"
stress_scale = 1.0e10

prop = Dict("name"=> testtype, "rho"=> 8000.0, "E"=> 200e+9, "nu"=> 0.45,
"sigmaY"=>0.3e+9, "K"=>1/9*200e+9,  "nn"=>post_nn)

include("NNPlatePull_Domain.jl")


# domain = Domain(nodes, elements, ndofs, EBC, g, FBC, fext)
@load "Data/domain.jld2" domain
state = zeros(domain.neqs)
∂u = zeros(domain.neqs)
globdat = GlobalData(state,zeros(domain.neqs), zeros(domain.neqs),∂u, domain.neqs, gt, ft)
assembleMassMatrix!(globdat, domain)

T = 0.0005
NT = 20
Δt = T/NT

nntype = "ae_scaled"
state_history, fext_history = read_data("$(@__DIR__)/Data/1.dat")
loss = DynamicMatLawLoss(domain, globdat, state_history, fext_history, nn,Δt)

sess = Session(); init(sess)
# ADCME.load(sess, "$(@__DIR__)/Data/learned_nn.mat")
# ADCME.load(sess, "Data/train_neural_network_from_fem.mat")
@show run(sess, loss)
# error()
BFGS!(sess, loss, 1000)

ADCME.save(sess, "Data/train_neural_network_from_fem.mat")

# * test neural network
close("all")
@load "Data/domain.jld2" domain
X, Y = prepare_strain_stress_data2D(domain)
x = constant(X)
y = nn(X[:,1:3], X[:,4:6], X[:,7:9])
O = run(sess, y)
using Random; Random.seed!(233)
VisualizeStress2D(Y, O, 20)

