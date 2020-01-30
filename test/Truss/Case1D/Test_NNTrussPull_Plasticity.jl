using Revise
using Test 
using NNFEM
using PyCall
using PyPlot
using JLD2
using ADCME
using LinearAlgebra


nnname = "Data/trained_nn_fem.mat"
s = ae_to_code(nnname, nntype)
eval(Meta.parse(s))
include("nnutil.jl")


# testtype = "PlaneStressPlasticity"
testtype = "NeuralNetwork1D"
include("NNTrussPull_Domain.jl")


prop = Dict("name"=> testtype, "rho"=> 0.1, "E"=> 200, "B"=> 10.0,
            "sigmaY"=>0.300, "K"=>1/9*200, "A0"=> 1.0, "eta"=> 10.0, "nn"=>post_nn)

prop = Dict("name"=> testtype, "rho"=> 8000.0, "E"=> 200e9, "nu"=> 0.45,
"sigmaY"=>0.3e9, "K"=>1/9*200e9, "B"=> 0.0, "A0"=> 1.0, "nn"=>post_nn)
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


T = 0.005
NT = 100
Δt = T/NT
for i = 1:NT
    @info i, "/" , NT
    solver = NewmarkSolver(Δt, globdat, domain, -1.0, 0.0, 1e-5, 1e-5, 100) # ok
end


domain_te = domain 
@info tid
@save "Data/domain_te$tid.jld2" domain_te