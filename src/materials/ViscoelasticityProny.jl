export  PlaneStrainViscoelasticityProny, PlaneStressViscoelasticityProny, ViscoelasticityProny

@doc raw"""
    mutable struct ViscoelasticityProny
        ρ::Float64 # density
        E::Float64 # Young's Modulus
        ν::Float64 # Poisson's ratio 
        τ::Float64 # relaxation time 
        c::Float64 # coefficient
        planestress::Bool 
        g0::Float64
        g1::Float64
        J::Float64
        σ0::Array{Float64} # stress at last time step
        σ0_::Array{Float64} # σ0 to be updated in `commitHistory`
        ε0::Array{Float64} # strain at last time step
        ε0_::Array{Float64} # ε0 to be updated in `commitHistory`
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
    g0::Float64
    g1::Float64
    J::Array{Float64,1}
    J_::Array{Float64,1}
    H::Array{Float64, 2}
    σ0::Array{Float64} # stress at last time step
    σ0_::Array{Float64} # σ0 to be updated in `commitHistory`
    ε0::Array{Float64} # strain at last time step
    ε0_::Array{Float64} # ε0 to be updated in `commitHistory`
end


function ViscoelasticityProny(prop::Dict{String, Any})
    ρ = Float64(prop["rho"])
    E = Float64(prop["E"])
    ν = Float64(prop["nu"])
    τ = Float64(prop["tau"])
    c = Float64(prop["c"])
    planestress = Bool(prop["planestress"])
    H = zeros(3,3)
    if planestress
        H[1,1] = E/(1. -ν*ν)
        H[1,2] = H[1,1]*ν
        H[2,1] = H[1,2]
        H[2,2] = H[1,1]
        H[3,3] = E/(2.0*(1.0+ν))
    else 
        H[1,1] = E*(1. -ν)/((1+ν)*(1. -2. *ν));
        H[1,2] = H[1,1]*ν/(1-ν);
        H[2,1] = H[1,2];
        H[2,2] = H[1,1];
        H[3,3] = H[1,1]*0.5*(1. -2. *ν)/(1. -ν);
    end

    planestress = Bool(prop["planestress"])
    σ0 = zeros(3); σ0_ = zeros(3)
    ε0 = zeros(3); ε0_ = zeros(3)
    J = zeros(3);  J_ = zeros(3)
    G0 = E/(2(1+ν))
    g0 = G0*(1-c)
    g1 = G0*c
    ViscoelasticityProny(ρ, E, ν, τ, c, planestress, g0, g1, J, J_, H, σ0, σ0_, ε0, ε0_)
end


function getStress(mat::ViscoelasticityProny,  strain::Array{Float64},  Dstrain::Array{Float64}, Δt::Float64)
    τ = mat.τ
    g0, g1, J = mat.g0, mat.g1, mat.J 
    σ = mat.H * strain - 2*g0*Dstrain + 2*g1*(- Dstrain) + exp(-Δt/τ)*J
    J = exp(-Δt/τ) * J + 2*g1*(strain - Dstrain)
    dΔσdΔε = mat.H 
    mat.σ0_ = σ 
    mat.ε0_ = strain 
    mat.J_ = J
    return σ, dΔσdΔε
end

function getTangent(mat::ViscoelasticityProny)
    error("Not implemented")
end

function commitHistory(mat::ViscoelasticityProny)
    mat.σ0 = mat.σ0_
    mat.ε0 = mat.ε0_
    mat.J = mat.J_
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