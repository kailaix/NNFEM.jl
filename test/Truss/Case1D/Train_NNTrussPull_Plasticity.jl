#= 


=#


using Revise
using Test 
using NNFEM
using PyCall
using PyPlot
using JLD2
using ADCME
using LinearAlgebra
reset_default_graph()
include("nnutil.jl")

testtype = "NeuralNetwork1D"
include("NNTrussPull_Domain.jl")

prop = Dict("name"=> testtype, "rho"=> 0.1, "E"=> 200.0, "B"=> 10.0,
            "sigmaY"=>0.300, "K"=>1/9*200, "A0"=> 1.0, "eta"=> 10.0, "nn"=>post_nn)
elements = []
for i = 1:nx 
    elnodes = [i, i+1]; coords = nodes[elnodes,:];
    push!(elements, FiniteStrainTruss(coords,elnodes, prop, ngp))
end
# domain = Domain(nodes, elements, ndofs, EBC, g, FBC, fext)
@load "Data/domain.jld2" domain
state = zeros(domain.neqs)
∂u = zeros(domain.neqs)
globdat = GlobalData(state,zeros(domain.neqs), zeros(domain.neqs),∂u, domain.neqs, gt, ft)
assembleMassMatrix!(globdat, domain)


T = 0.5
NT = 20
Δt = T/NT


nntype = "ae_scaled"


E = prop["E"]
H0 = zeros(1,1)
H0[1,1] = E

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
# BFGS!(sess, loss, 800)

X, Y = prepare_strain_stress_data(domain)
y = squeeze(nn(constant(X[:,1]),constant(X[:,2]),constant(X[:,3])))
close("all")
out = run(sess, y)
plot(X[:,1], out,"+", label="NN")
plot(X[:,1], Y, ".", label="Exact")
legend()

