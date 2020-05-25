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
    σ0::Float64 # stress at last time step
    σ0_::Float64 # σ0 to be updated in `commitHistory
end


function ViscoelasticityMaxwell(prop::Dict{String, Any})
    ρ = prop["rho"]
    E = prop["E"]
    ν = prop["nu"]
    η = prop["eta"]
    λ = E*ν/(1+ν)/(1-2ν)
    μ = E/(2(1+ν))
    σ0 = zeros(3); σ0_ = zeros(3)
    ϵ0 = zeros(3); ϵ0_ = zeros(3)
    ViscoelasticityMaxwell(ρ, E, ν, η, λ, μ, σ0, σ0_, ϵ0, ϵ0_)
end



function getStress(domain::ViscoelasticityMaxwell,  strain::Float64,  Dstrain::Float64, Δt::Float64)
    
    local dΔσdΔε
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
    return σ, dΔσdΔε
end

function getTangent(domain::ViscoelasticityMaxwell)
    error("Not implemented")
end

function commitHistory(domain::ViscoelasticityMaxwell)
    domain.σ0 = domain.σ0_
end
