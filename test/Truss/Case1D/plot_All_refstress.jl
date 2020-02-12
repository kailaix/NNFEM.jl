using PyPlot
using PyCall
#mpl = pyimport("tikzplotlib")
using JLD2


close("all")
for tid = 1:5
    @show tid
@load "Data/domain$(tid).jld2" domain



strain = hcat(domain.history["strain"]...)
stress = hcat(domain.history["stress"]...)
i=8
markevery=5
plot(strain[i,1:markevery:end], stress[i,1:markevery:end], "--+", label="Reference-tid$(tid)")

end

xlabel("Strain")
ylabel("Stress (MPa)")
legend()
#mpl.save("truss1d_stress$tid.tex")
savefig("all_truss1d_strain_stress.pdf")