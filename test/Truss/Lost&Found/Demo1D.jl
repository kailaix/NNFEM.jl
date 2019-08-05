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
# # # need to update state in domain from globdat
# updateStates!(domain, globdat)

# # # # for debug
# X, Y = prepare_strain_stress_data(domain)
# x = constant(X)
# y = squeeze(ae(x, [20,20,20,20,1], "nn"))
# sess = Session(); init(sess)
# close("all")
# ADCME.load(sess, "Data/learned_nn.mat")
# out = run(sess, y)
# plot(X[:,2], out,"+", label="NN")
# plot(X[:,2], Y, ".", label="Exact")
# legend()


T = 0.5
NT = 20
Δt = T/NT


nntype = "ae_scaled"
E = prop["E"]
H0 = zeros(1,1)
H0[1,1] = E




state_history, fext_history = read_data("$(@__DIR__)/Data/1.dat")
loss = DynamicMatLawLossTest(domain, globdat, state_history, fext_history, nn,Δt)


variable_scope(nntype) do
    global opt = AdamOptimizer().minimize(loss)
end
sess = Session(); init(sess)
# ADCME.load(sess, "Data/learned_nn.mat")
@show run(sess, loss)

for i = 1:2000
    l, _ = run(sess, [loss, opt])
    @show i,l
end

X, Y = prepare_strain_stress_data(domain)
x = constant([X[:,1] X[:,2] X[:,3]/100])
y = squeeze(ae(x, [20,20,20,20,1], nntype)*100.0)
close("all")
out = run(sess, y)
plot(X[:,2], out,"+", label="NN")
plot(X[:,2], Y, ".", label="Exact")
legend()
