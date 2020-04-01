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

testtype = "NeuralNetwork1D"

prop = Dict("name"=> testtype, "rho"=> 0.1, "E"=> 200.0, "B"=> 10.0,
            "sigmaY"=>0.300, "K"=>1/9*200, "A0"=> 1.0, "eta"=> 10.0, "nn"=>post_nn)

include("NNTrussPull_Domain.jl")

# domain = Domain(nodes, elements, ndofs, EBC, g, FBC, fext)
@load "Data/domain1.jld2" domain
state = zeros(domain.neqs)
∂u = zeros(domain.neqs)
globdat = GlobalData(state,zeros(domain.neqs), zeros(domain.neqs),∂u, domain.neqs, gt, ft)
assembleMassMatrix!(globdat, domain)

nntype = "ae_scaled"
# nntype = "linear"
n_data = [1,2,4,5]
losses = Array{PyObject}(undef, length(n_data))
for (i, ni) in enumerate(n_data)
    state_history, fext_history = read_data("$(@__DIR__)/Data/$i.dat")
    losses[i] = DynamicMatLawLoss(domain, globdat, state_history, fext_history, nn,Δt)
end
loss = sum(losses)

sess = Session(); init(sess)
@show run(sess, loss)
BFGS!(sess, loss, 2000)
ADCME.save(sess, "Data/trained_nn_fem.mat")
