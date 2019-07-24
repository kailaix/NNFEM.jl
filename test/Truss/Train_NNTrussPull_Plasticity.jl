using Revise
using Test 
using NNFEM
using PyCall
using PyPlot
using JLD2
using ADCME
using LinearAlgebra

include("nnutil.jl")

testtype = "NeuralNetwork1D"
include("NNTrussPull_Domain.jl")

prop = Dict("name"=> testtype, "rho"=> 0.1, "E"=> 200, "B"=> 10.0,
            "sigmaY"=>0.300, "K"=>1/9*200, "A0"=> 1.0, "eta"=> 10.0, "nn"=>post_nn)
elements = []
for i = 1:nx 
    elnodes = [i, i+1]; coords = nodes[elnodes,:];
    push!(elements, FiniteStrainTruss(coords,elnodes, prop, ngp))
end
domain = Domain(nodes, elements, ndofs, EBC, g, FBC, fext)
state = zeros(domain.neqs)
∂u = zeros(domain.neqs)
globdat = GlobalData(state,zeros(domain.neqs), zeros(domain.neqs),∂u, domain.neqs, gt, ft)
assembleMassMatrix!(globdat, domain)
# need to update state in domain from globdat
updateStates!(domain, globdat)


T = 0.5
NT = 20
Δt = T/NT


nntype = "nn"


E = prop["E"]
H0 = zeros(1,1)
H0[1,1] = E


W1 = Variable(rand(3,3))
b1 = Variable(rand(3))
W2 = Variable(rand(3,3))
b2 = Variable(rand(3))
W3 = Variable(rand(3,1))
b3 = Variable(rand(1))

_W1 = Variable(rand(3,1))
_b1 = Variable(rand(1))
_W2 = Variable(rand(3,1))
_b2 = Variable(rand(1))
_W3 = Variable(rand(3,1))
_b3 = Variable(rand(1))

all_vars = [W1,b1,W2,b2,W3,b3,_W1,_b1,_W2,_b2,_W3,_b3]


n_data = 1
losses = Array{PyObject}(undef, n_data)
for i = 1:n_data
    state_history, fext_history = read_data("$(@__DIR__)/Data/$i.dat")
    losses[i] = DynamicMatLawLoss(domain, globdat, state_history, fext_history, nn,Δt)
end
loss = sum(losses)
sess = Session(); init(sess)
@show run(sess, loss)
BFGS(sess, loss)
println("Real H = ", H0)


# save nn parameters
var_val = run(sess, all_vars)
for i = 1:length(all_vars)
    if !isdir("$(@__DIR__)/Data/Weights")
        mkdir("$(@__DIR__)/Data/Weights")
    end
    writedlm("$(@__DIR__)/Data/Weights/$i.txt", var_val[i])
end