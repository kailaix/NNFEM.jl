export  Viscoplasticity1D

mutable struct Viscoplasticity1D
    ρ::Float64 # density
    E::Float64 # Young's modulus
    # hardening parameter, yield function = |σ - q| - (σY + Kα)
    K::Float64
    B::Float64
    η::Float64 
    σY::Float64
    α::Float64 
    α_::Float64 # α to be updated in `commitHistory`
    σ0::Float64 # stress at last time step
    σ0_::Float64 # σ0 to be updated in `commitHistory
    q::Float64 # stress at last time step
    q_::Float64 # σ0 to be updated in `commitHistory`
end


function Viscoplasticity1D(prop::Dict{String, Any})
    ρ = prop["rho"]; 
    E = prop["E"]; B = prop["B"]; 
    K = prop["K"]; σY = prop["sigmaY"]
    η = prop["eta"]
    α = 0.0; α_ = 0.0
    σ0 = 0.0; σ0_ = 0.0
    q = 0.0; q_ = 0.0
    Viscoplasticity1D(ρ, E, K, B, η, σY, α, α_, σ0, σ0_, q, q_)
end



@doc """
    For Viscoplasticity material
    :param hysteresis_variables: [eps_vp, alpha, q], plastic strain, internal hardening variable, and back stress
    The yield condition is
        f = |sigma - q| - (σY + alpha * K)
    here K is the plastic modulus, , σY is the flow stress
    D_eps_vp = <phi(f)>/eta * df/dsigma
    D_alpha = |D_eps_p|
    D_q     = B * D_eps_vp

    here <phi(f)> = (f+|f|)/2

""" -> 
function getStress(self::Viscoplasticity1D,  strain::Float64,  Dstrain::Float64, Δt::Float64)
    
    local dΔσdΔε
    ε = strain; ε0 = Dstrain 
    σ0 = self.σ0;  α0 = self.α;  q0 = self.q
    E = self.E;    K = self.K;   B = self.B; 
    η = self.η; σY = self.σY 
    
    #trial stress
    Δγ = 0.0
    σ = σ0 + E*(ε-ε0) 
    α = α0 + abs(Δγ)
    q = q0 + B*Δγ
    ξ = σ - q

    r2 = abs(ξ) - (σY + K*α)
    if r2 <= 0
        σ = σ0 + E*(ε-ε0)
        dΔσdΔε = E

    else

        Δγ = r2 * Δt / (η + Δt * (E + H + B))
        q += B * Δγ * sign(ξ)
        α += Δγ

        σ -= Δγ * E * sign(ξ)
        dΔσdΔε = E * (η + Δt*(B + K)) / (η + Δt*(B + E + K))
    end
            
    # #@show Δγ
    self.α_  = self.α + Δγ
    self.σ0_ = σ
    self.q_  = q

    return σ, dΔσdΔε
end

function getTangent(self::Viscoplasticity1D)
    error("Not implemented")
end

function commitHistory(self::Viscoplasticity1D)
    self.α = self.α_
    self.σ0 = self.σ0_
    self.q = self.q_
end
