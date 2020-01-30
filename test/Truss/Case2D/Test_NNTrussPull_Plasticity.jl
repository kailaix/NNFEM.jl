using Revise
using Test 
using NNFEM
using PyCall
using PyPlot
using JLD2
using ADCME
using LinearAlgebra
using MAT


nnname = "Data/trained_nn_fem.mat"
nntype = "ae_scaled"
s = ae_to_code(nnname, nntype)

eval(Meta.parse(s))
include("nnutil.jl")

# testtype = "PlaneStressPlasticity"
testtype = "NeuralNetwork1D"

prop = Dict("name"=> testtype, "rho"=> 0.1, "E"=> 200, "B"=> 10.0,
            "sigmaY"=>0.300, "K"=>1/9*200, "A0"=> 1.0, "eta"=> 10.0, "nn"=>post_nn)

include("NNTrussPull_Domain.jl")

domain = Domain(nodes, elements, ndofs, EBC, g, FBC, fext)
state = zeros(domain.neqs)
∂u = zeros(domain.neqs)
globdat = GlobalData(state,zeros(domain.neqs), zeros(domain.neqs),∂u, domain.neqs, gt, ft)
assembleMassMatrix!(globdat, domain)
# need to update state in domain from globdat
updateStates!(domain, globdat)

for i = 1:NT
    @info i, "/" , NT
    solver = NewmarkSolver(Δt, globdat, domain, -1.0, 0.0, 1e-5, 1e-5, 100) # ok

end
@info tid
domain_te = domain
@save "Data/domain$(tid)_te.jld2" domain_te


