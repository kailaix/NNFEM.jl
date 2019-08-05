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
include("NNTrussPull_Domain.jl")

prop = Dict("name"=> testtype, "rho"=> 0.1, "E"=> 200.0, "B"=> 10.0,
            "sigmaY"=>0.300, "K"=>1/9*200, "A0"=> 1.0, "eta"=> 10.0, "nn"=>post_nn)
elements = []
for i = 1:nx 
    elnodes = [i, i+1]; coords = nodes[elnodes,:];
    push!(elements, FiniteStrainTruss(coords,elnodes, prop, ngp))
end
# domain = Domain(nodes, elements, ndofs, EBC, g, FBC, fext)
@load "Data/domain.jld2" domain
state = zeros(domain.neqs)
∂u = zeros(domain.neqs)
globdat = GlobalData(state,zeros(domain.neqs), zeros(domain.neqs),∂u, domain.neqs, gt, ft)
assembleMassMatrix!(globdat, domain)
# # need to update state in domain from globdat
# updateStates!(domain, globdat)


T = 0.5
NT = 20
Δt = T/NT


nntype = "ae"
E = prop["E"]
H0 = zeros(1,1)
H0[1,1] = E


state_history, fext_history = read_data("$(@__DIR__)/Data/1.dat")
loss = DynamicMatLawLossTest(domain, globdat, state_history, fext_history, nn,Δt)

sess = Session(); init(sess)
@show run(sess, loss)
