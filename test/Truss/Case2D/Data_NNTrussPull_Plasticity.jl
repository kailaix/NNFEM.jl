using Revise
using Test 
using NNFEM
using PyCall
using PyPlot
using JLD2
using ADCME
using LinearAlgebra

tid = 1

# testtype = "PathDependent1D" 
testtype = "Plasticity1D"

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

for i = 1:NT
    solver = NewmarkSolver(Δt, globdat, domain, -1.0, 0.0, 1e-5, 100)
    # close("all")
    # scatter(nodes[:, 1], nodes[:,2], color="red")
    # u,v = domain.state[1:domain.nnodes], domain.state[domain.nnodes+1:end]
    # scatter(nodes[:, 1] + u, nodes[:,2] + v, color="blue")
    # savefig("$(@__DIR__)/Debug/$i.png")

end
# error()
# error()
# todo write data
write_data("$(@__DIR__)/Data/$tid.dat", domain)
# plot
close("all")
scatter(nodes[:, 1], nodes[:,2], color="red")
u,v = domain.state[1:domain.nnodes], domain.state[domain.nnodes+1:end]
scatter(nodes[:, 1] + u, nodes[:,2] + v, color="blue")
ylim(-0.2,1.5)
savefig("$(@__DIR__)/Debug/terminal$tid.png")
# error()
@save "Data/domain.jld2" domain


X, Y = prepare_strain_stress_data1D(domain)
close("all")
plot(X[:,1], Y, ".", label="Exact")
grid("on")
legend()
xlim(-0.05,0.05)
ylim(-2,2)
savefig("$(@__DIR__)/Debug/law$tid.png")



