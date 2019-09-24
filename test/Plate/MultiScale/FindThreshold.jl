include("nnutil.jl")
using PyPlot
for force_scale in [4.0,5.0,6.0,50.0]
    close("all")
    for k in [100, 200, 201, 202, 203, 300]
        @info force_scale, k
        @load "Data/order2/domain$(k)_$(force_scale)_5.jld2" domain
        M = []
        for v in domain.history["stress"]
            m = maximum(sum(v.^2, dims=2))
            push!(M, m)
        end
        plot(M, label="case$k")
    end
    grid("on")
    legend()
    title("force_scale=$force_scale")
    savefig("Debug/threshold/$force_scale.png")
end
