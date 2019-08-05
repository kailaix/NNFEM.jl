using Revise
using Test 
using NNFEM
using PyCall
using PyPlot
using JLD2
using ADCME
using LinearAlgebra

include("nnutil.jl")

testtype = "NeuralNetwork2D"
include("NNPlatePull_Domain.jl")


prop = Dict("name"=> testtype, "rho"=> 8000.0, "E"=> 200e+9, "nu"=> 0.45,
"sigmaY"=>0.3e+9, "K"=>1/9*200e+9, "nn"=>post_nn)

elements = []
for j = 1:ny
    for i = 1:nx 
        n = (nx+1)*(j-1) + i
        elnodes = [n, n + 1, n + 1 + (nx + 1), n + (nx + 1)]
        coords = nodes[elnodes,:]
        push!(elements,SmallStrainContinuum(coords,elnodes, prop,2))
    end
end


domain = Domain(nodes, elements, ndofs, EBC, g, NBC, fext)
state = zeros(domain.neqs)
∂u = zeros(domain.neqs)
globdat = GlobalData(state,zeros(domain.neqs),
                    zeros(domain.neqs),∂u, domain.neqs, gt)

assembleMassMatrix!(globdat, domain)
updateStates!(domain, globdat)


T = 2.0
NT = 20
Δt = T/NT

nntype = "ae_scaled"

n_data = 1
losses = Array{PyObject}(undef, n_data)
for i = 1:n_data
    state_history, fext_history = read_data("$(@__DIR__)/Data/$i.dat")
    losses[i] = DynamicMatLawLoss(domain, globdat, state_history, fext_history, nn,Δt)
end
loss = sum(losses)

variable_scope(nntype) do
    global opt = AdamOptimizer().minimize(loss)
end

sess = Session(); init(sess)
@show run(sess, loss)

for i = 1:2000
    l, _ = run(sess, [loss, opt])
    @show i,l
    # if l<20000
    #     break
    # end
end

ADCME.save(sess, "Data/trained_nn_fem.mat")
