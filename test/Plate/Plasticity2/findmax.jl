include("nnutil.jl")

for k in [100, 201, 202, 203]
    @load "Data/order2/domain$(k)_50.0_10.jld2" domain
    M = []
    for v in domain.history["strain"]
        m = maximum(sum(v.^2, dims=2))
        push!(M, m)
    end
    @show maximum(M)
end
