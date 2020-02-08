using Revise
using Test 
using NNFEM
using PyCall
using PyPlot
using JLD2
using ADCME
using LinearAlgebra



include("NNTrussPull_Domain.jl")
testtype = "Plasticity1D" 

# kg, m,  ms
prop = Dict("name"=> testtype, "rho"=> 8000.0, "E"=> 200e3, "nu"=> 0.45,
           "sigmaY"=>0.3e3, "K"=>1/9*200e3, "B"=> 0.0, "A0"=> 0.005)


           

nodes, EBC, g, gt, FBC, fext, ft, npoints, node_to_point = BoundaryCondition(tid)
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



for i = 1:NT
    solver = NewmarkSolver(Δt, globdat, domain, -1.0, 0.0, 1e-8, 1e-8, 100) # ok
    printstyled("============================== iteration $i ==============================\n",color=:green)
end

if !isdir("$(@__DIR__)/Data/")
    mkdir("$(@__DIR__)/Data/")
end

write_data("$(@__DIR__)/Data/$tid.dat", domain)
@save "$(@__DIR__)/Data/domain$tid.jld2" domain
# plot
# close("all")
# scatter(nodes[:, 1], nodes[:,2], color="red")
# u,v = domain.state[1:domain.nnodes], domain.state[domain.nnodes+1:end]
# scatter(nodes[:, 1] + u, nodes[:,2] + v, color="blue")

# @save "Data/domain.jld2" domain