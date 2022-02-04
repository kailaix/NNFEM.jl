using Revise
using Test 
using NNFEM
using PyCall
using PyPlot
using JLD2
using ADCME
using LinearAlgebra


include("Truss_Domain.jl")
testtype = "Elasticity1D" 

tid = 4

# kg, m,  ms
# E = 200e9 Pa / 1e6
s_scale = 1e6
prop = Dict("name"=> testtype, "rho"=> 8000.0, "E"=> 200e3, "A0"=> 0.005)


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



SolverInitial!(Δt, globdat, domain)

ω = EigenMode(Δt, globdat, domain)
@show "stable time step is ", 0.8 * 2/ω, " current time step is ", Δt
for i = 1:NT
    @info i, "/" , NT
    solver = ExplicitSolver(Δt, globdat, domain)
    if i%10 == 0
        ω = EigenMode(Δt, globdat, domain)
        @show "stable time step is ", 0.8 * 2/ω, " current time step is ", Δt
    end
end

if !isdir("$(@__DIR__)/Data/")
    mkdir("$(@__DIR__)/Data/")
end


# plot
close("all")
strain = hcat(domain.history["strain"]...)
stress = hcat(domain.history["stress"]...)
sid = 4 #Gaussian point id
plot(strain[sid,:], stress[sid,:]/s_scale)
savefig("Debug/ref/strain-stress$(tid).png") 

close("all")
ts = LinRange(0,T, NT+1)

ux = hcat(domain.history["state"]...)[1:nx+1, :]
nid = 3 #node id
plot(ts, ux[nid,:])
savefig("Debug/ref/disp$(tid).png") 

# @save "Data/domain.jld2" domain