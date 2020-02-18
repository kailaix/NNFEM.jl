using PyPlot
using PyCall
mpl = pyimport("tikzplotlib")
using JLD2

rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
font0 = Dict(
        "font.size" => 16,
        "axes.labelsize" => 16,
        "xtick.labelsize" => 16,
        "ytick.labelsize" => 16,
        "legend.fontsize" => 16,
)
merge!(rcParams, font0)


close("all")
for tid = 1:5
    @show tid
@load "Data/domain$(tid).jld2" domain



strain = hcat(domain.history["strain"]...)
stress = hcat(domain.history["stress"]...)
i=8
markevery=5
plot(strain[i,1:markevery:end], stress[i,1:markevery:end], "--o", fillstyle="none", label="Reference-tid$(tid)")

end

xlabel("Strain")
ylabel("Stress (MPa)")
legend()
PyPlot.tight_layout()
savefig("all_truss1d_strain_stress.pdf")
mpl.save("all_truss1d_strain_stress.tex")
