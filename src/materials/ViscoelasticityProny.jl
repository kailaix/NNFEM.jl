export  PlaneStrainViscoelasticityProny, PlaneStressViscoelasticityProny, ViscoelasticityProny

@doc raw"""
    mutable struct ViscoelasticityProny
        ρ::Float64 # density
        E::Float64 # Young's Modulus
        ν::Float64 # Poisson's ratio 
        τ::Float64 # relaxation time 
        c::Float64 # coefficient
        planestress::Bool 
        σ0::Array{Float64} # stress at last time step
        σ0_::Array{Float64} # σ0 to be updated in `commitHistory`
        ε0::Array{Float64} # strain at last time step
        ε0_::Array{Float64} # ε0 to be updated in `commitHistory`
        t0::Float64 # relaxation timer
        t0_::Float64 # t0 to be updated 
    end

Prony series viscoelasticity model.

$$E(t) = E_0 - cE_0(1-\exp(-t/\tau))$$

In principle, we can have different Prony series for both shear modulus $G$ and bulk modulus $K$, 

$$\sigma_{ij} = 3K \left( \frac{1}{3} \epsilon_{kk}\delta_{ij}\right) + 2G \left( \epsilon_{ij} - \frac{1}{3}\epsilon_{kk} \delta_{ij} \right)$$

We only implemented $E(t)$ case for simplicity. 
"""
mutable struct ViscoelasticityProny
    ρ::Float64 # density
    E::Float64 # Young's Modulus
    ν::Float64 # Poisson's ratio 
    τ::Float64 # relaxation time 
    c::Float64 # coefficient
    planestress::Bool 
    G0::Float64 # Shear Modulus
    σ0::Array{Float64} # stress at last time step
    σ0_::Array{Float64} # σ0 to be updated in `commitHistory`
    ε0::Array{Float64} # strain at last time step
    ε0_::Array{Float64} # ε0 to be updated in `commitHistory`
    t0::Float64 # relaxation timer
    t0_::Float64 # t0 to be updated 
end


function ViscoelasticityProny(prop::Dict{String, Any})
    ρ = Float64(prop["rho"])
    E = Float64(prop["E"])
    ν = Float64(prop["nu"])
    τ = Float64(prop["tau"])
    c = Float64(prop["c"])
    G0 = E/(2*(1+ν))
    planestress = Bool(prop["planestress"])
    σ0 = zeros(3); σ0_ = zeros(3)
    ε0 = zeros(3); ε0_ = zeros(3)
    t0 = t0_ = 0.0
    ViscoelasticityProny(ρ, E, ν, τ, c, planestress, G0, σ0, σ0_, ε0, ε0_, t0, t0_)
end


function getStress(mat::ViscoelasticityProny,  strain::Array{Float64},  Dstrain::Array{Float64}, Δt::Float64)
    E = mat.E 
    ν = mat.ν
    t = mat.t0 
    τ = mat.τ
    c = mat.c 
    G0 = mat.G0 
    G = G0 * (1.0 - c * ( 1- exp(-t/τ)))
    E = G/G0 * E
    H = zeros(3,3)
    if mat.planestress
        H[1,1] = E/(1. -ν*ν)
        H[1,2] = H[1,1]*ν
        H[2,1] = H[1,2]
        H[2,2] = H[1,1]
        H[3,3] = G
    else 
        H[1,1] = E*(1. -ν)/((1+ν)*(1. -2. *ν));
        H[1,2] = H[1,1]*ν/(1-ν);
        H[2,1] = H[1,2];
        H[2,2] = H[1,1];
        H[3,3] = H[1,1]*0.5*(1. -2. *ν)/(1. -ν);
    end
    σ = H * (strain - mat.ε0) + mat.σ0
    dΔσdΔε = H 
    mat.σ0_ = σ
    mat.ε0_ = strain
    mat.t0_ = mat.t0 + Δt
    return σ, dΔσdΔε
end

function getTangent(mat::ViscoelasticityProny)
    error("Not implemented")
end

function commitHistory(mat::ViscoelasticityProny)
    mat.σ0 = mat.σ0_
    mat.ε0 = mat.ε0_
    mat.t0 = mat.t0_
end

"""
    PlaneStrainViscoelasticityProny(prop::Dict{String, Any})

Returns the plane strain viscoelasticity Prony shear series material model.
"""
function PlaneStrainViscoelasticityProny(prop::Dict{String, Any})
    prop["planestress"] = false
    ViscoelasticityProny(prop)
end

"""
    PlaneStrainViscoelasticityProny(prop::Dict{String, Any})

Returns the plane stress viscoelasticity Prony shear series material model.
"""
function PlaneStressViscoelasticityProny(prop::Dict{String, Any})
    prop["planestress"] = true
    ViscoelasticityProny(prop)
end