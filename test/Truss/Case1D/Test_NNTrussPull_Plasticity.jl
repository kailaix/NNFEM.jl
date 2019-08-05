using Revise
using Test 
using NNFEM
using PyCall
using PyPlot
using JLD2
using ADCME
using LinearAlgebra

include("nnutil.jl")


# testtype = "PlaneStressPlasticity"
testtype = "NeuralNetwork1D"
include("NNTrussPull_Domain.jl")


prop = Dict("name"=> testtype, "rho"=> 0.1, "E"=> 200, "B"=> 10.0,
            "sigmaY"=>0.300, "K"=>1/9*200, "A0"=> 1.0, "eta"=> 10.0, "nn"=>post_nn)
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


T = 0.5
NT = 20
Δt = T/NT
for i = 1:NT
    @info i, "/" , NT
    solver = NewmarkSolver(Δt, globdat, domain, -1.0, 0.0, 1e-5, 100)
end


close("all")
scatter(nodes[:, 1], nodes[:,2], color="red")
u,v = domain.state[1:domain.nnodes], domain.state[domain.nnodes+1:end]
scatter(nodes[:, 1] + u, nodes[:,2] + v, color="blue")



@load "Data/domain.jld2" domain
close("all")
X, Y = prepare_strain_stress_data(domain)
y = zeros(size(X,1))
for i = 1:length(y)
    y[i] = post_nn(X[i,1], X[i,2], X[i,3], Δt)[1]
end
close("all")
plot(X[:,1], y,"+", label="NN")
plot(X[:,1], Y, ".", label="Exact")
legend()


# # * test gradients
# ε0 = rand()
# σ0 = rand()
# ε = rand()
# Δ = rand()

# z0, g0 = post_nn(ε, ε0, σ0, Δt)
# sval_ = zeros(5)
# wval_ = zeros(5)
# gs_ = zeros(5)
# for i = 1:5
#     γ = 1/2^i
#     gs_[i] = γ
#     z = post_nn(ε+γ*Δ, ε0, σ0, Δt)
#     sval_[i] = z[1]-z0
#     wval_[i] = z[1]-z0-g0*Δ*γ
# end
# close("all")
# loglog(gs_, abs.(sval_), "*-", label="finite difference")
# loglog(gs_, abs.(wval_), "+-", label="derivative")
# loglog(gs_, gs_.^2 * 0.5*abs(wval_[1])/gs_[1]^2, "--",label="\$\\mathcal{O}(\\gamma^2)\$")
# loglog(gs_, gs_ * 0.5*abs(sval_[1])/gs_[1], "--",label="\$\\mathcal{O}(\\gamma)\$")

# plt.gca().invert_xaxis()
# legend()
# xlabel("\$\\gamma\$")
# ylabel("Error")
