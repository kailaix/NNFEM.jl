export  Elasticity1D
mutable struct Elasticity1D
    E::Float64
    ρ::Float64
    σ0::Float64 # stress at last time step
    σ0_::Float64 # σ0 to be updated in `commitHistory`
    ε0::Float64
    ε0_::Float64
end

function Elasticity1D(prop::Dict{String, Any})
    E = prop["E"]; ρ = prop["rho"]
    σ0 = 0.0; σ0_ = 0.0; ε0 = 0.0; ε0_ = 0.0
    Elasticity1D(E, ρ, σ0, σ0_, ε0, ε0_)
end

function getStress(self::Elasticity1D, strain::Float64, Dstrain::Float64, Δt::Float64 = 0.0)
    sigma = self.E * strain
    self.σ0_ = sigma
    self.ε0_ = strain

    return sigma, self.E
end

function getTangent(self::Elasticity1D)
    self.E
end

function commitHistory(self::Elasticity1D)
    self.σ0 = self.σ0_
    self.ε0 = self.ε0_
end
