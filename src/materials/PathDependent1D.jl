export  PathDependent1D

mutable struct PathDependent1D
    ρ::Float64 # density
    E::Float64 # Young's modulus
    # hardening parameter, yield function = |σ - q| - (σY + Kα)
    K::Float64
    B::Float64 
    σY::Float64
    α::Float64 
    α_::Float64 # α to be updated in `commitHistory`
    σ0::Float64 # stress at last time step
    σ0_::Float64 # σ0 to be updated in `commitHistory`
    q::Float64 # stress at last time step
    q_::Float64 # σ0 to be updated in `commitHistory`
    ε0::Float64 # strain at last time step
    ε0_::Float64 # ε0 to be updated in `commitHistory`
end


function PathDependent1D(prop::Dict{String, Any})
    ρ = prop["rho"]; 
    E = prop["E"]; B = prop["B"]; 
    K = prop["K"]; σY = prop["sigmaY"]
    α = 0.0; α_ = 0.0
    σ0 = 0.0; σ0_ = 0.0
    q = 0.0; q_ = 0.0
    ε0 = 0.0; ε0_ = 0.0
    PathDependent1D(ρ, E, K, B, σY, α, α_, σ0, σ0_, q, q_, ε0, ε0_)
end



@doc """
    For debugging pathdependent materials, we an build arbitrary pathdependent material law

    sigma = sigma0 + (eps - eps0)**2

""" -> 
function getStress(self::PathDependent1D,  strain::Float64,  Dstrain::Float64, Δt::Float64 = 0.0)
    
    local dΔσdΔε
    ε = strain; ε0 = Dstrain 
    σ0 = self.σ0;  α0 = self.α;  q0 = self.q
    E = self.E;    K = self.K;   B = self.B; 
    σY = self.σY 
    
    σ = σ0 + (ε - ε0)*(ε - ε0)/2.0
    dΔσdΔε = (ε - ε0)
    
    self.α_  = self.α
    self.q_  = self.q

    self.σ0_ = σ
    self.ε0_ = ε

    return σ, dΔσdΔε
end

function getTangent(self::PathDependent1D)
    error("Not implemented")
end

function commitHistory(self::PathDependent1D)
    self.α = self.α_
    self.σ0 = self.σ0_
    self.q = self.q_
    self.ε0 = self.ε0_
end
