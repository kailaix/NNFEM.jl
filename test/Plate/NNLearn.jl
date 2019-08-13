using Revise
using ADCME
using NNFEM
using JLD2
using LinearAlgebra
reset_default_graph()

include("nnutil.jl")
stress_scale = 1.0e10
strain_scale = 1.0

nntype = "mae"
H0 = Variable(diagm(0=>ones(3)))
ndata = 1

loss = constant(0.0)
for i = 1:ndata
    global loss
    @load "Data/domain$i.jld2" domain
    X, Y = prepare_strain_stress_data2D(domain)
    x = constant(X)
    y = nn(X[:,1:3], X[:,4:6], X[:,7:9])

    loss += sum((y-Y)^2)
end
variable_scope(nntype) do
    global opt = AdamOptimizer().minimize(loss)
end

sess = Session(); init(sess)
@show run(sess, loss)
ADCME.load(sess, "Data/learned_nn.mat")
BFGS!(sess, loss, 20)
ADCME.save(sess, "Data/learned_nn.mat")

error()
close("all")
@load "Data/domain1.jld2" domain
X, Y = prepare_strain_stress_data2D(domain)
x = constant(X)
y = nn(X[:,1:3], X[:,4:6], X[:,7:9])

init(sess)
# ADCME.load(sess, "Data/learned_nn.mat")
O = run(sess, y)

using Random; Random.seed!(233)
VisualizeStress2D(Y, O, 20)
# ADCME.save(sess, "Data/learned_nn.mat")

error("Learning stop!")

ADCME.load(sess, "Data/learned_nn.mat")
@show run(sess, loss)
close("all")
O = run(sess, y)
using Random; Random.seed!(233)
VisualizeStress2D(Y, O, 20)



VisualizeStrainStressSurface(X, Y)
VisualizeStrainStressSurface(X, O)
