using Revise
using ADCME
using NNFEM
using JLD2
using PyPlot
reset_default_graph()

include("nnutil.jl")

loss = constant(0.0)
for tid = [1,2,4,5]
    strain, stress = read_strain_stress("Data/$(tid).dat")
    X, Y = prepare_strain_stress_data1D(strain, stress )

    #yy = E0*(X[:,1] - X[:,2]) + X[:,3]

    #@show (Y - X[:,3]) ./ (X[:,1] - X[:,2])
    y = squeeze(nn(constant(X[:,1:1]), constant(X[:,2:2]), constant(X[:,3:3])))
    global loss += mean((y - Y)^2)
end
sess = Session(); init(sess)
@show run(sess,loss)

if !isdir("Data/$(nntype)")
    mkdir("Data/$(nntype)")
end

for i = 1:10
    println("************************** Outer Iteration = $i ************************** ")
    BFGS!(sess, loss, 1000)
    ADCME.save(sess, "Data/$(nntype)/learned_nn_ite$(i).mat")
end


# error()

tid = 3
strain, stress = read_strain_stress("Data/$(tid).dat")
X, Y = prepare_strain_stress_data1D(strain, stress )
y = squeeze(nn(constant(X[:,1:1]), constant(X[:,2:2]), constant(X[:,3:3])))
# sess = Session(); init(sess)
# close("all")
# ADCME.load(sess, "Data/$(nntype)/learned_nn_ite5.mat")
out = run(sess, y)
plot(X[:,1], out,"+", label="NN")
plot(X[:,1], Y, ".", label="Exact")
#legend()
savefig("nnlearn_$(nntype)_truss1d_stress$tid.png")



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

