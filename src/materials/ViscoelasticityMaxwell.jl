export  ViscoelasticityMaxwell

@doc raw"""
    mutable struct ViscoelasticityMaxwell
        ρ::Float64 # density
        E::Float64 # Young's modulus
        ν::Float64 # Poisson's ratio
        η::Float64 # Viscosity parameter
        λ::Float64
        μ::Float64 # Lame constants
        σ0::Float64 # stress at last time step
        σ0_::Float64 # σ0 to be updated in `commitHistory
    end

Maxwell model for viscoelasticity. For details, see [this post](https://kailaix.github.io/PoreFlow.jl/dev/viscoelasticity/)

The updating rule is 

$$\sigma^{n+1} = H \epsilon^{n+1} + S \sigma^n  - H\epsilon^n$$

"""
mutable struct ViscoelasticityMaxwell
    ρ::Float64 # density
    E::Float64 # Young's modulus
    ν::Float64 # Poisson's ratio 
    η::Float64 # Viscosity parameter
    λ::Float64
    μ::Float64 # Lame constants
    σ0::Array{Float64} # stress at last time step
    σ0_::Array{Float64} # σ0 to be updated in `commitHistory`
    ε0::Array{Float64} # strain at last time step
    ε0_::Array{Float64} # ε0 to be updated in `commitHistory`
end


function ViscoelasticityMaxwell(prop::Dict{String, Any})
    ρ = Float64(prop["rho"])
    E = Float64(prop["E"])
    ν = Float64(prop["nu"])
    η = Float64(prop["eta"])
    λ = E*ν/(1+ν)/(1-2ν)
    μ = E/(2(1+ν))
    σ0 = zeros(3); σ0_ = zeros(3)
    ε0 = zeros(3); ε0_ = zeros(3)
    ViscoelasticityMaxwell(ρ, E, ν, η, λ, μ, σ0, σ0_, ε0, ε0_)
end



function getStress(domain::ViscoelasticityMaxwell,  strain::Array{Float64},  Dstrain::Array{Float64}, Δt::Float64)
    # strain, Dstrain are defined at (1-αf)*Δt in the generalized α solver 
    ϵ = strain; ϵ0 = Dstrain
    σ0 = domain.σ0 
    λ, μ, η = domain.λ, domain.μ, domain.η
    S = inv([
        1 + 2/3*μ*Δt/η -1/3*μ*Δt/η 0.0 
        -1/3*μ*Δt/η 1+2/3*μ*Δt/η 0.0
        0.0 0.0 1+μ*Δt/η
    ])
    H = S * [2μ+λ λ 0.0 
            λ 2μ+λ 0.0 
            0.0 0.0 μ]
    σ = H * ϵ + S * σ0 - H * ϵ0
    dΔσdΔε = H 
    domain.σ0_ = σ
    domain.ε0_ = ϵ
    return σ, dΔσdΔε
end

function getTangent(domain::ViscoelasticityMaxwell)
    error("Not implemented")
end

function commitHistory(domain::ViscoelasticityMaxwell)
    domain.σ0 = domain.σ0_
    domain.ε0 = domain.ε0_
end
