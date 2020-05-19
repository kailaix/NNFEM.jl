export  Scalar1D
mutable struct Scalar1D
    κ::Float64
end

function Scalar1D(prop::Dict{String, Any})
    κ = prop["kappa"]
    Scalar1D(κ)
end

function getStress(mat::Scalar1D, ∇u::Array{Float64}, ∇u_t::Array{Float64}, Δt::Float64 = 0.0)
    sigma = mat.κ * ∇u 
    H = mat.κ * diagm(0=>ones(2))
    return sigma, H
end
