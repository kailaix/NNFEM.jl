using PyPlot
using PyCall
mpl = pyimport("tikzplotlib")
using JLD2
using NNFEM
include("../nnutil.jl")
nntype= "ae_scaled"
stress_scale = 100.0


tid = 3
@load "../Data/domain$tid.jld2" domain 
@load "../Data/learn_domain$(tid)_te.jld2" domain_te

u1 = hcat(domain.history["state"]...)
u2 = hcat(domain_te.history["state"]...)
err = zeros(8, 101)
for i = 1:8
    u1_ = [u1[i,:] u1[i+8,:]]
    u2_ = [u2[i,:] u2[i+8,:]]
    err[i,:] = sqrt.(sum((u1_-u2_).^2,dims=2)[:])
end

T = 0.5
NT = 100
t = LinRange(0.0,T, NT+1)
for i = 1:8
    plot(t, err[i,:])
end
xlabel("t")
ylabel("\$u_{ref}-u_{est}\$")
mpl.save("truss2d_learn_loc_diff$tid.tex")


@load "Data/domain3.jld2" domain
X, Y = prepare_strain_stress_data1D(domain)
y = squeeze(nn(constant(X[:,1]), constant(X[:,2]), constant(X[:,3])))
sess = Session(); init(sess)
close("all")
ADCME.load(sess, "Data/learned_nn.mat")
out = run(sess, y)
plot(X[:,1], out,"+", label="NN")
plot(X[:,1], Y, ".", label="Exact")
legend()
