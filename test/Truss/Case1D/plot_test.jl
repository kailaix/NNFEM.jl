using PyPlot
using PyCall
#mpl = pyimport("tikzplotlib")
using JLD2


T = 200.0
NT = 200
t = LinRange(0,T,NT+1)

tid = length(ARGS)>=2 ? parse(Int64, ARGS[2]) : 3
nntype = length(ARGS)>=1 ? ARGS[1] : "piecewise"

@load "Data/domain$tid.jld2" domain 
@load "Data/$(nntype)/domain_te$(tid).jld2" domain_te 
#domain_te = domain

# close("all")
# u1 = hcat(domain.history["state"]...)
# u2 = hcat(domain_te.history["state"]...)
# u = abs.(u1 - u2)
# for i = 1:5
#     plot(t, u[i,:], label="$i")
# end
# xlabel("\$t\$")
# ylabel("\$||u_{ref}-u_{exact}||\$")
# legend()
# #mpl.save("truss1d_disp_diff$tid.tex")
# savefig("truss1d_disp_diff.pdf")

close("all")
strain = hcat(domain.history["strain"]...)
stress = hcat(domain.history["stress"]...)
i = 8
plot(strain[i,:], stress[i,:], "--", label="Reference")


strain = hcat(domain_te.history["strain"]...)
stress = hcat(domain_te.history["stress"]...)
i = 8
plot(strain[i,:], stress[i,:], ".", label="Estimated")

xlabel("Strain")
ylabel("Stress")
legend()
#mpl.save("truss1d_stress$tid.tex")
savefig("nn_$(nntype)_truss1d_stress$tid.pdf")