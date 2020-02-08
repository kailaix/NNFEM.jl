using Revise
using ADCME
using NNFEM
using JLD2
using PyPlot

include("nnutil.jl")

loss = constant(0.0)
for tid = [1,2,4,5]
    @load "Data/domain$tid.jld2" domain
    X, Y = prepare_strain_stress_data1D(domain)
    y = squeeze(nn(constant(X[:,1]), constant(X[:,2]), constant(X[:,3])))
    global loss += sum((y-Y)^2)
end
sess = Session(); init(sess)
@show run(sess,loss)

if !isdir("Data/$(nntype)")
    mkdir("Data/$(nntype)")
end

for i = 1:5
    println("************************** Outer Iteration = $i ************************** ")
    BFGS!(sess, loss, 1000)
    ADCME.save(sess, "Data/$(nntype)/learned_nn_ite$(i).mat")
end


# error()

# @load "Data/domain3.jld2" domain
# X, Y = prepare_strain_stress_data1D(domain)
# x = constant(X)
# y = squeeze(nn(constant(X[:,1]), constant(X[:,2]), constant(X[:,3])))
# sess = Session(); init(sess)
# close("all")
# ADCME.load(sess, "Data/learned_nn.mat")
# out = run(sess, y)
# plot(X[:,1], out,"+", label="NN")
# plot(X[:,1], Y, ".", label="Exact")
# legend()


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

