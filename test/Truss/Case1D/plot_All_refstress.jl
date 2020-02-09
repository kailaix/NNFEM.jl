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
plot(strain[i,:], stress[i,:], "--", label="Reference$(tid)-$(i)")

end

xlabel("Strain")
ylabel("Stress")
legend()
#mpl.save("truss1d_stress$tid.tex")
savefig("all_truss1d_stress.pdf")