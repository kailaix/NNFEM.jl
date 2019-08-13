using Revise
using Test 
using NNFEM
using PyCall
using PyPlot
using JLD2
using ADCME
using MAT
using LinearAlgebra

tid = 1
include("nnutil.jl")
nntype = "maeadd"


# * Auto-generated code by `ae_to_code`
aedictae_scaled = matread("Data/learned_nn.mat"); # using MAT
# aedictae_scaled = matread("Data/train_neural_network_from_fem.mat")
Wkey = "$(nntype)backslashfully_connectedbackslashweightscolon0"
Wkey = "$(nntype)backslashfully_connected_1backslashweightscolon0"
Wkey = "$(nntype)backslashfully_connected_2backslashweightscolon0"
Wkey = "$(nntype)backslashfully_connected_3backslashweightscolon0"
Wkey = "$(nntype)backslashfully_connected_4backslashweightscolon0"
function nnae_scaled(net)
        # @show size(net)
        # return reshape(reshape(net[1:3],1,3)*H0,1,3)/stress_scale
        W0 = aedictae_scaled["$(nntype)backslashfully_connectedbackslashweightscolon0"]; b0 = aedictae_scaled["$(nntype)backslashfully_connectedbackslashbiasescolon0"];
        isa(net, Array) ? (net = net * W0 .+ b0') : (net = net *W0 + b0)
        isa(net, Array) ? (net = tanh.(net)) : (net=tanh(net))
        W1 = aedictae_scaled["$(nntype)backslashfully_connected_1backslashweightscolon0"]; b1 = aedictae_scaled["$(nntype)backslashfully_connected_1backslashbiasescolon0"];
        isa(net, Array) ? (net = net * W1 .+ b1') : (net = net *W1 + b1)
        isa(net, Array) ? (net = tanh.(net)) : (net=tanh(net))
        W2 = aedictae_scaled["$(nntype)backslashfully_connected_2backslashweightscolon0"]; b2 = aedictae_scaled["$(nntype)backslashfully_connected_2backslashbiasescolon0"];
        isa(net, Array) ? (net = net * W2 .+ b2') : (net = net *W2 + b2)
        isa(net, Array) ? (net = tanh.(net)) : (net=tanh(net))
        W3 = aedictae_scaled["$(nntype)backslashfully_connected_3backslashweightscolon0"]; b3 = aedictae_scaled["$(nntype)backslashfully_connected_3backslashbiasescolon0"];
        isa(net, Array) ? (net = net * W3 .+ b3') : (net = net *W3 + b3)
        isa(net, Array) ? (net = tanh.(net)) : (net=tanh(net))
        W4 = aedictae_scaled["$(nntype)backslashfully_connected_4backslashweightscolon0"]; b4 = aedictae_scaled["$(nntype)backslashfully_connected_4backslashbiasescolon0"];
        isa(net, Array) ? (net = net * W4 .+ b4') : (net = net *W4 + b4)
        return net
end


# testtype = "PlaneStressPlasticity"
testtype = "NeuralNetwork2D"
prop = Dict("name"=> testtype, "rho"=> 8000.0, "E"=> 200e+9, "nu"=> 0.45,
"sigmaY"=>0.3e+9, "K"=>1/9*200e+9,  "nn"=>post_nn)
include("NNPlatePull_Domain.jl")



domain = Domain(nodes, elements, ndofs, EBC, g, FBC, fext)
state = zeros(domain.neqs)
∂u = zeros(domain.neqs)
globdat = GlobalData(state,zeros(domain.neqs), zeros(domain.neqs),∂u, domain.neqs, gt, ft)

assembleMassMatrix!(globdat, domain)
updateStates!(domain, globdat)

for i = 1:NT
    @info i, "/" , NT
    solver = NewmarkSolver(Δt, globdat, domain, -1.0, 0.0, 1e-3, 100)
    
end

close("all")
visσ(domain)
savefig("Debug/$(tid)_test.png")

# visstatic(domain)