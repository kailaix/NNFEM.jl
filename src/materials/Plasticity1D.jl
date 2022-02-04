export  Plasticity1D

mutable struct Plasticity1D
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


function Plasticity1D(prop::Dict{String, Any})
    ρ = prop["rho"]; 
    E = prop["E"]; B = prop["B"]; 
    K = prop["K"]; σY = prop["sigmaY"]
    α = 0.0; α_ = 0.0
    σ0 = 0.0; σ0_ = 0.0
    q = 0.0; q_ = 0.0
    ε0 = 0.0; ε0_ = 0.0
    Plasticity1D(ρ, E, K, B, σY, α, α_, σ0, σ0_, q, q_, ε0, ε0_)
end



@doc """
    For Plasticity material
    :param hysteresis_variables: [eps_vp, alpha, q], plastic strain, internal hardening variable, and back stress
    The yield condition is
        f = |sigma - q| - (σY + alpha * K)
    here K is the plastic modulus, , σY is the flow stress
    D_eps_p = gamma df/dsigma        
    D_alpha = |D_eps_p|
    D_q     = B * D_eps_p
    f  = 0    or f  < 0

""" -> 
function getStress(self::Plasticity1D,  strain::Float64,  Dstrain::Float64, Δt::Float64 = 0.0)
    
    local dΔσdΔε
    ε = strain; ε0 = Dstrain 
    σ0 = self.σ0;  α0 = self.α;  q0 = self.q
    E = self.E;    K = self.K;   B = self.B; 
    σY = self.σY 
    
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
        @info "Plasticity"
        Δγ = r2/(B + E + K)
        q += B * Δγ * sign(ξ)
        α += Δγ

        σ -= Δγ * E * sign(ξ)
        dΔσdΔε = E*(B + K)/(B + E + K)
    end
            
    # @info "Plasticity1D ", ε, ε0, self.σ0, σ
    self.α_  = self.α + Δγ
    self.σ0_ = σ
    self.q_  = q
    self.ε0_ = ε

    return σ, dΔσdΔε
end

function getTangent(self::Plasticity1D)
    error("Not implemented")
end

function commitHistory(self::Plasticity1D)
    self.α = self.α_
    self.σ0 = self.σ0_
    self.q = self.q_
    self.ε0 = self.ε0_
end
