using Revise
using ADCME
using NNFEM
using JLD2
using PyPlot

include("nnutil.jl")
nntype= "ae_scaled"
stress_scale = 100.0
@load "Data/domain.jld2" domain

X, Y = prepare_strain_stress_data1D(domain)
y = squeeze(nn(constant(X[:,1]), constant(X[:,2]), constant(X[:,3])))
loss = sum((y-Y)^2)
variable_scope("nn") do
    global opt = AdamOptimizer().minimize(loss)
end

sess = Session(); init(sess)
@show run(sess,loss)
# for i = 1:50000
#     l, _ = run(sess, [loss, opt])
#     @show i,l
# end
BFGS!(sess, loss, 1000)
out = run(sess, y)
close("all")
plot(X[:,2], out,"+", label="NN")
plot(X[:,2], Y, ".", label="Exact")
legend()

ADCME.save(sess, "Data/learned_nn.mat")
error()

@load "Data/domain.jld2" domain
X, Y = prepare_strain_stress_data1D(domain)
x = constant(X)
y = squeeze(nn(constant(X[:,1]), constant(X[:,2]), constant(X[:,3])))
sess = Session(); init(sess)
close("all")
ADCME.load(sess, "Data/learned_nn.mat")
out = run(sess, y)
plot(X[:,1], out,"+", label="NN")
plot(X[:,1], Y, ".", label="Exact")
legend()


# @load "Data/domain.jld2" domain
# X, Y = prepare_strain_stress_data1D(domain)
# x = constant(X)
# y = squeeze(nn(constant(X[:,1]), constant(X[:,2]), constant(X[:,3])))
# sess = Session(); init(sess)
# close("all")
# ADCME.load(sess, "Data/learned_nn.mat")
# out = run(sess, y)
# plot(X[:,1], out,"+", label="NN")
# plot(X[:,1], Y, ".", label="Exact")
# ADCME.load(sess, "Data/trained_nn_fem.mat")
# out = run(sess, y)
# plot(X[:,1], out,">", label="End-to-end")
# legend()

