using Revise
using Test 
using NNFEM
using PyCall
using PyPlot
using JLD2
using ADCME
using LinearAlgebra

include("NNTrussPull_Domain.jl")
testtype = "Plasticity1D" #"Elasticity1D"
#testtype = "Elasticity1D"





prop = Dict("name"=> testtype, "rho"=> 0.1, "E"=> 200, "B"=> 10.0,
            "sigmaY"=>0.300, "K"=>1/9*200, "A0"=> 1.0, "eta"=> 10.0)


elements = []
for i = 1:nx 
    elnodes = [i, i+1]; coords = nodes[elnodes,:];
    push!(elements, FiniteStrainTruss(coords,elnodes, prop, ngp))

end



domain = Domain(nodes, elements, ndofs, EBC, g, NBC, fext)
state = zeros(domain.neqs)
∂u = zeros(domain.neqs)
globdat = GlobalData(state,zeros(domain.neqs),
                    zeros(domain.neqs),∂u, domain.neqs, gt)

assembleMassMatrix!(globdat, domain)
updateStates!(domain, globdat)


T = 1.0
NT = 20
Δt = T/NT
for i = 1:NT
    @info i, "/" , NT
    solver = NewmarkSolver(Δt, globdat, domain, -1.0, 0.0, 1e-5, 100)
    
end

# error()
# todo write data
write_data("$(@__DIR__)/Data/1.dat", domain)
# plot
scatter(nodes[:, 1], nodes[:,2], color="red")
u,v = domain.state[1:domain.nnodes], domain.state[domain.nnodes+1:end]
scatter(nodes[:, 1] + u, nodes[:,2] + v, color="blue")