using Revise
using ADCME
using NNFEM
using JLD2
reset_default_graph()
include("nnutil.jl")

nntype = "ae_scaled"
@load "Data/domain.jld2" domain

stress_scale = 1.0e10
X, Y = prepare_strain_stress_data2D(domain)
x = constant(X)
y = nn(X[:,1:3], X[:,4:6], X[:,7:9])

loss = sum((y-Y)^2)/stress_scale^2
variable_scope(nntype) do
    global opt = AdamOptimizer().minimize(loss)
end

sess = Session(); init(sess)
@show run(sess, loss)

# error()
# for i = 1:2000
#     l, _ = run(sess, [loss, opt])
#     @show i,l
# end
BFGS!(sess, loss, 15000)
# ADCME.load(sess, "Data/learned_nn.mat")
# @show run(sess, loss)
close("all")
O = run(sess, y)
using Random; Random.seed!(233)
VisualizeStress2D(Y, O, 20)
ADCME.save(sess, "Data/learned_nn.mat")
error("Learning stop!")

ADCME.load(sess, "Data/learned_nn.mat")
@show run(sess, loss)
close("all")
O = run(sess, y)
using Random; Random.seed!(233)
VisualizeStress2D(Y, O, 20)
