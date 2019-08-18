using Revise
using Test 
using NNFEM
using PyCall
using PyPlot
using JLD2
using ADCME
using LinearAlgebra
# reset_default_graph()
include("nnutil.jl")

testtype = "NeuralNetwork2D"
nntype = "ae_scaled"
# H0 = SPDMatrix(3)
H0 = Variable(rand(3,3))
n_data = 5


prop = Dict("name"=> testtype, "rho"=> 800.0, "E"=> 200e+9, "nu"=> 0.45,
"sigmaY"=>0.3e+9, "K"=>1/9*200e+9,  "nn"=>post_nn)
ps = PlaneStress(prop); H0 = ps.H
# H0 = constant(HH)/stress_scale

include("NNPlatePull_Domain.jl")


# domain = Domain(nodes, elements, ndofs, EBC, g, FBC, fext)
@load "Data/domain1.jld2" domain
state = zeros(domain.neqs)
∂u = zeros(domain.neqs)
globdat = GlobalData(state,zeros(domain.neqs), zeros(domain.neqs),∂u, domain.neqs, gt, ft)
assembleMassMatrix!(globdat, domain)

losses = Array{PyObject}(undef, n_data)
for i = 1:n_data
    state_history, fext_history = read_data("$(@__DIR__)/Data/$i.dat")
    losses[i] = DynamicMatLawLoss(domain, globdat, state_history, fext_history, nn,Δt)
end
loss = sum(losses)/stress_scale^2

sess = Session(); init(sess)
# ADCME.load(sess, "$(@__DIR__)/Data/learned_nn.mat")
ADCME.load(sess, "Data/train_neural_network_from_fem.mat")
# @show run(sess, loss)
# error()
BFGS!(sess, loss, 1500)
# ADCME.save(sess, "$(@__DIR__)/Data/train_neural_network_from_fem.mat")
# ADCME.load(sess, "$(@__DIR__)/Data/train_neural_network_from_fem.mat")
# BFGS!(sess, loss, 5000)
# ADCME.save(sess, "$(@__DIR__)/Data/train_neural_network_from_fem.mat")
# BFGS!(sess, loss, 5000)
ADCME.save(sess, "$(@__DIR__)/Data/train_neural_network_from_fem.mat")

# * test neural network
close("all")
@load "$(@__DIR__)/Data/domain1.jld2" domain
X, Y = prepare_strain_stress_data2D(domain)
x = constant(X)
y = nn(X[:,1:3], X[:,4:6], X[:,7:9])
init(sess)
ADCME.load(sess, "$(@__DIR__)/Data/train_neural_network_from_fem.mat")

try
    global O = run(sess, y)
catch
    global O = y 
end
using Random; Random.seed!(233)
VisualizeStress2D(Y, O, 20)
savefig("$(@__DIR__)/Debug/trained_nn.png")
