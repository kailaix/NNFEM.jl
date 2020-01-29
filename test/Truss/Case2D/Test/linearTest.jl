using Revise
using Test 
using NNFEM
using PyCall
using PyPlot
using JLD2
using ADCME
using LinearAlgebra

tid = 1

testtype = "Elasticity1D" 

prop = Dict("name"=> testtype, "rho"=> 0.1, "E"=> 200, "B"=> 10.0,
            "sigmaY"=>0.300, "K"=>1/9*200, "A0"=> 1.0, "eta"=> 10.0)

include("NNTrussPull_Domain.jl")


domain = Domain(nodes, elements, ndofs, EBC, g, FBC, fext)
state = zeros(domain.neqs)
∂u = zeros(domain.neqs)
globdat = GlobalData(state,zeros(domain.neqs), zeros(domain.neqs),∂u, domain.neqs, gt, ft)
assembleMassMatrix!(globdat, domain)

# need to update state in domain from globdat
updateStates!(domain, globdat)


Fint, _ = assembleStiffAndForce(globdat, domain, Δt)
# tfAssembleInternalForce(domain, nn, E_all, DE_all, w∂E∂u_all, σ0_all)


