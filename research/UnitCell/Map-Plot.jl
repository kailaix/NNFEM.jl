using NPZ
using LinearAlgebra
using PyPlot
rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
    mysize = 15
    font0 = Dict(
    "font.size" => mysize,
    "axes.labelsize" => mysize,
    "xtick.labelsize" => mysize,
    "ytick.labelsize" => mysize,
    "legend.fontsize" => mysize,
    )
merge!(rcParams, font0)

using Random, Distributions
include("PlatePull.jl")
include("Plot.jl")


seed = 123
Random.seed!(seed)
N_data = 10000
porder = 2
N_θ = 100
θ_field = rand(Normal(0, 1.0), N_data, N_θ);


L = 1
N_x = 1000
xx = LinRange(0, L, N_x) 
pn = zeros(Float64, N_x)


for ind in [1,11]
    for i = 1:N_x
        pn[i] = 100 + 400.0*fxθ(xx[i], θ_field[ind, :])
    end
    fig = figure()
    plot(xx, pn, "-r")
    savefig("Fig/Solid-map-input-$(ind).png")

    

    fig = figure()
    GenerateData(θ_field[ind, :], porder; plot=true, plotname="Solid-map-output-$(ind)")
    
    
end
