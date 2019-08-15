using Revise
using Test 
using NNFEM
using PyCall
using PyPlot
using JLD2
using ADCME
using MAT
using LinearAlgebra

include("nnutil.jl")

nntype = "ae_scaled"
tid = 3.5

# * Auto-generated code by `ae_to_code`
if nntype=="ae_scaled"
#      global aedictae_scaled = matread("$(@__DIR__)/Data/learned_nn.mat"); # using MAT
     global aedictae_scaled = matread("Data/train_neural_network_from_fem.mat")
end
Wkey = "ae_scaledbackslashfully_connectedbackslashweightscolon0"
Wkey = "ae_scaledbackslashfully_connected_1backslashweightscolon0"
Wkey = "ae_scaledbackslashfully_connected_2backslashweightscolon0"
Wkey = "ae_scaledbackslashfully_connected_3backslashweightscolon0"
Wkey = "ae_scaledbackslashfully_connected_4backslashweightscolon0"
function nnae_scaled(net)
        W0 = aedictae_scaled["ae_scaledbackslashfully_connectedbackslashweightscolon0"]; b0 = aedictae_scaled["ae_scaledbackslashfully_connectedbackslashbiasescolon0"];
        isa(net, Array) ? (net = net * W0 .+ b0') : (net = net *W0 + b0)
        isa(net, Array) ? (net = tanh.(net)) : (net=tanh(net))
        W1 = aedictae_scaled["ae_scaledbackslashfully_connected_1backslashweightscolon0"]; b1 = aedictae_scaled["ae_scaledbackslashfully_connected_1backslashbiasescolon0"];
        isa(net, Array) ? (net = net * W1 .+ b1') : (net = net *W1 + b1)
        isa(net, Array) ? (net = tanh.(net)) : (net=tanh(net))
        W2 = aedictae_scaled["ae_scaledbackslashfully_connected_2backslashweightscolon0"]; b2 = aedictae_scaled["ae_scaledbackslashfully_connected_2backslashbiasescolon0"];
        isa(net, Array) ? (net = net * W2 .+ b2') : (net = net *W2 + b2)
        isa(net, Array) ? (net = tanh.(net)) : (net=tanh(net))
        W3 = aedictae_scaled["ae_scaledbackslashfully_connected_3backslashweightscolon0"]; b3 = aedictae_scaled["ae_scaledbackslashfully_connected_3backslashbiasescolon0"];
        isa(net, Array) ? (net = net * W3 .+ b3') : (net = net *W3 + b3)
        isa(net, Array) ? (net = tanh.(net)) : (net=tanh(net))
        W4 = aedictae_scaled["ae_scaledbackslashfully_connected_4backslashweightscolon0"]; b4 = aedictae_scaled["ae_scaledbackslashfully_connected_4backslashbiasescolon0"];
        isa(net, Array) ? (net = net * W4 .+ b4') : (net = net *W4 + b4)
        return net
end


# testtype = "PlaneStressPlasticity"
testtype = "NeuralNetwork2D"
prop = Dict("name"=> testtype, "rho"=> 8000.0, "E"=> 200e+9, "nu"=> 0.45,
"sigmaY"=>0.3e+9, "K"=>1/9*200e+9, "nn"=>post_nn)
include("NNPlatePull_Domain.jl")

@load "Data/domain.jld2" domain
stress_scale = 1.0e10
X, Y = prepare_strain_stress_data2D(domain)
x = constant(X)
y = nn(X[:,1:3], X[:,4:6], X[:,7:9])

sess = Session(); init(sess)
# ADCME.load(sess, "Data/learned_nn.mat")
ADCME.load(sess, "Data/train_neural_network_from_fem.mat")

O = run(sess, y)
using Random; Random.seed!(233)
VisualizeStress2D(Y, O, 20)
