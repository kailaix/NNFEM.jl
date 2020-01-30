using PyPlot
using PyCall
mpl = pyimport("tikzplotlib")
using JLD2

tid = 1

@load "../Data/domain$tid.jld2" domain 
@load "../Data/domain_te$tid.jld2" domain_te 

close("all")
T = 0.005
NT = 100
t = LinRange(0,T,NT+1)
u1 = hcat(domain.history["state"]...)
u2 = hcat(domain_te.history["state"]...)

u = abs.(u1 - u2)
for i = 1:5
    plot(t, u[i,:], label="$i")
end
xlabel("\$t\$")
ylabel("\$||u_{ref}-u_{exact}||\$")
legend()
mpl.save("truss1d_disp_diff$tid.tex")
# savefig("truss1d_disp_diff.pdf")

close("all")
T = 0.005
NT = 100
t = LinRange(0,T,NT+1)
strain = hcat(domain.history["strain"]...)
stress = hcat(domain.history["stress"]...)
i = 8
plot(strain[i,:], stress[i,:], "--", label="Reference")


strain = hcat(domain.history["strain"]...)
stress = hcat(domain.history["stress"]...)
i = 8
plot(strain[i,:], stress[i,:], ".", label="Estimated")

xlabel("Strain")
ylabel("Stress")
legend()
mpl.save("truss1d_stress$tid.tex")
