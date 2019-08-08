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

prop = Dict("name"=> testtype, "rho"=> 0.1, "E"=> 200.0, "B"=> 10.0,
            "sigmaY"=>0.300, "K"=>1/9*200, "A0"=> 1.0, "eta"=> 10.0, "nn"=>post_nn)

include("NNTrussPull_Domain.jl")

# domain = Domain(nodes, elements, ndofs, EBC, g, FBC, fext)
@load "Data/domain.jld2" domain
state = zeros(domain.neqs)
∂u = zeros(domain.neqs)
globdat = GlobalData(state,zeros(domain.neqs), zeros(domain.neqs),∂u, domain.neqs, gt, ft)
assembleMassMatrix!(globdat, domain)




# nntype = "ae_scaled"
nntype = "linear"
n_data = 1
losses = Array{PyObject}(undef, n_data)
for i = 1:n_data
    state_history, fext_history = read_data("$(@__DIR__)/Data/$i.dat")
    losses[i] = DynamicMatLawLoss(domain, globdat, state_history, fext_history, nn,Δt)
end
loss = sum(losses)

sess = Session(); init(sess)
@show run(sess, loss)
error()
# ADCME.load(sess,  "Data/learned_nn.mat")
# error()
BFGS!(sess, loss, 100)
ADCME.save(sess, "Data/trained_nn_fem.mat")

# # for online training
# for i = 1:10
#     BFGS!(sess, loss, 15000)
#     ADCME.save(sess, "Data/trained_nn_fem_$i.mat")
# end
# error()

# * test
@load "Data/domain.jld2" domain
X, Y = prepare_strain_stress_data1D(domain)
x = constant(X)
y = squeeze(nn(constant(X[:,1]), constant(X[:,2]), constant(X[:,3])))
sess = Session(); init(sess)
close("all")
ADCME.load(sess, "Data/trained_nn_fem.mat")
out = run(sess, y)
plot(X[:,1], out,"+", label="NN")
plot(X[:,1], Y, ".", label="Exact")
legend()
