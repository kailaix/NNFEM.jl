export  Elasticity1D
mutable struct Elasticity1D
    E::Float64
    ρ::Float64
end

function Elasticity1D(prop::Dict{String, Any})
    E = prop["E"]; ρ = prop["rho"]
    Elasticity1D(E, ρ)
end

function getStress(self::Elasticity1D, strain::Float64, Dstrain::Float64)
    sigma = self.E * strain

    return sigma, self.E
end

function getTangent(self::Elasticity1D)
    self.E
end

function commitHistory(self::Elasticity1D)
end
