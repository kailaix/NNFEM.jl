using Revise
using Test 
using NNFEM
using PyCall
using PyPlot
using JLD2
using ADCME
using LinearAlgebra


# testtype = "PathDependent1D" 
testtype = "Plasticity1D"

# prop = Dict("name"=> testtype, "rho"=> 0.1, "E"=> 200, "B"=> 10.0,
#             "sigmaY"=>0.300, "K"=>1/9*200, "A0"=> 1.0, "eta"=> 10.0)

prop = Dict("name"=> testtype, "rho"=> 8000.0, "E"=> 200e9, "nu"=> 0.45,
            "sigmaY"=>0.3e9, "K"=>1/9*200e9, "B"=> 0.0, "A0"=> 0.005)

include("NNTrussPull_Domain.jl")


domain = Domain(nodes, elements, ndofs, EBC, g, FBC, fext)
state = zeros(domain.neqs)
∂u = zeros(domain.neqs)
globdat = GlobalData(state,zeros(domain.neqs), zeros(domain.neqs),∂u, domain.neqs, gt, ft)
assembleMassMatrix!(globdat, domain)

# need to update state in domain from globdat
updateStates!(domain, globdat)

for i = 1:NT
    solver = NewmarkSolver(Δt, globdat, domain, -1.0, 0.0, 1e-5, 1e-5, 100) # ok
    # close("all")
    # scatter(nodes[:, 1], nodes[:,2], color="red")
    # u,v = domain.state[1:domain.nnodes], domain.state[domain.nnodes+1:end]
    # scatter(nodes[:, 1] + u, nodes[:,2] + v, color="blue")
    # savefig("$(@__DIR__)/Debug/$i.png")

end
# error()
# error()

if !isdir("$(@__DIR__)/Data")
    mkdir("$(@__DIR__)/Data")
end
if !isdir("$(@__DIR__)/Debug")
    mkdir("$(@__DIR__)/Debug")
end

write_data("$(@__DIR__)/Data/$tid.dat", domain)
@save "Data/domain$tid.jld2" domain
# close("all")
# scatter(nodes[:, 1], nodes[:,2], color="red")
# u,v = domain.state[1:domain.nnodes], domain.state[domain.nnodes+1:end]
# scatter(nodes[:, 1] + u, nodes[:,2] + v, color="blue")
# ylim(-0.2,1.5)
# savefig("$(@__DIR__)/Debug/terminal$tid.png")


