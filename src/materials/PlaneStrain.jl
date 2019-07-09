export  PlaneStrain
mutable struct PlaneStrain
    H::Array{Float64}
    E::Float64
    ν::Float64
    ρ::Float64
end

function PlaneStrain(prop::Dict{String, Any})
    E = prop["E"]; ν = prop["nu"]; ρ = prop["rho"]
    H = zeros(3,3)
    H[1,1] = E*(1. -ν)/((1+ν)*(1. -2. *ν));
    H[1,2] = H[1,1]*ν/(1-ν);
    H[2,1] = H[1,2];
    H[2,2] = H[1,1];
    H[3,3] = H[1,1]*0.5*(1. -2. *ν)/(1. -ν);
    PlaneStrain(H, E, ν, ρ)
end

function getStress(self::PlaneStrain, strain::Array{Float64}, Dstrain::Array{Float64}, Δt::Float64 = 0.0)
    sigma = self.H * strain

    return sigma, self.H
end

function getTangent(self::PlaneStrain)
    self.H
end

function commitHistory(self::PlaneStrain)
end
