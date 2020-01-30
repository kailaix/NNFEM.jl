using PyPlot
using PyCall
mpl = pyimport("tikzplotlib")
using JLD2
using NNFEM


tid = 1
@load "../Data/domain$tid.jld2" domain 
@load "../Data/domain$(tid)_te.jld2" domain_te

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
mpl.save("truss2d_loc_diff$tid.tex")

close("all")
X, Y = prepare_strain_stress_data1D(domain)
X_te, Y_te = prepare_strain_stress_data1D(domain_te)
scatter(X[:,1], Y[:], marker=".",s=5)
scatter(X_te[:,1], Y_te[:], marker=".",s=5)
xlabel("Strain")
ylabel("Stress")
mpl.save("truss2d_stress_diff$tid.tex")
