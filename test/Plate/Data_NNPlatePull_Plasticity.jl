using Revise
using Test 
using NNFEM
using PyCall
using PyPlot
using JLD2
using ADCME
using LinearAlgebra


testtype = "PlaneStressPlasticity"

prop = Dict("name"=> testtype, "rho"=> 8000.0, "E"=> 200e+9, "nu"=> 0.45,
"sigmaY"=>0.3e+9, "K"=>1/9*200e+9)

include("NNPlatePull_Domain.jl")


domain = Domain(nodes, elements, ndofs, EBC, g, FBC, fext)
state = zeros(domain.neqs)
∂u = zeros(domain.neqs)
globdat = GlobalData(state,zeros(domain.neqs), zeros(domain.neqs),∂u, domain.neqs, gt, ft)

assembleMassMatrix!(globdat, domain)
updateStates!(domain, globdat)


T = 0.0005
NT = 20
Δt = T/NT
for i = 1:NT
    # @info i, "/" , NT
    solver = NewmarkSolver(Δt, globdat, domain, -1.0, 0.0, 1e-4, 100)
    close("all")
    visσ(domain,-1.5e9, 4.5e9)
    savefig("Debug/$i.png")
    # error()
end

# error()
# todo write data
write_data("$(@__DIR__)/Data/1.dat", domain)
# plot
close("all")
scatter(nodes[:, 1], nodes[:,2], color="red")
u,v = domain.state[1:domain.nnodes], domain.state[domain.nnodes+1:end]
scatter(nodes[:, 1] + u, nodes[:,2] + v, color="blue")

@save "Data/domain.jld2" domain