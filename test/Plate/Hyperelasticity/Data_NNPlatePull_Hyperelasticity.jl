using Revise
using Test 
using NNFEM
using PyCall
using PyPlot
using JLD2
using ADCME
using LinearAlgebra

tid = 3.5
printstyled("tid = $tid\n", color=:cyan)
testtype = "PlaneStressIncompressibleRivlinSaunders"
#https://classes.engineering.wustl.edu/2009/spring/mase5513/abaqus/docs/v6.6/books/bmk/default.htm?startat=ch01s01ach08.html
prop = Dict("name"=> testtype, "rho"=> 800.0, "C1"=>0.1863e6, "C2"=>-0.00979e6)

# testtype = "PlaneStress"
# prop = Dict("name"=> testtype, "rho"=> 8000.0, "E"=> 200e+9, "nu"=> 0.45)

include("NNPlatePull_Domain.jl")


domain = Domain(nodes, elements, ndofs, EBC, g, FBC, fext)
state = zeros(domain.neqs)
∂u = zeros(domain.neqs)
globdat = GlobalData(state,zeros(domain.neqs), zeros(domain.neqs),∂u, domain.neqs, gt, ft)

assembleMassMatrix!(globdat, domain)
updateStates!(domain, globdat)


for i = 1:NT
    @info i, "/" , NT
    solver = NewmarkSolver(Δt, globdat, domain, -1.0, 0.0, 1e-4, 100)
    # close("all")
    # visσ(domain,-1.5e9, 4.5e9)
    # savefig("Debug/$i.png")
    # error()
    if i==75
        close("all")
        visσ(domain)
        # visσ(domain,-1.5e9, 4.5e9)
        savefig("Debug/terminal$(tid)i=75.png")
    end
end

# error()
# todo write data
write_data("$(@__DIR__)/Data/$tid.dat", domain)
# plot
close("all")
scatter(nodes[:, 1], nodes[:,2], color="red")
u,v = domain.state[1:domain.nnodes], domain.state[domain.nnodes+1:end]
scatter(nodes[:, 1] + u, nodes[:,2] + v, color="blue")

close("all")
visσ(domain)
# visσ(domain,-1.5e9, 4.5e9)
savefig("Debug/terminal$tid.png")

@save "Data/domain$tid.jld2" domain