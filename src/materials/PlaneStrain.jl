export  PlaneStrain
mutable struct PlaneStrain
    H::Array{Float64}
    E::Float64
    ν::Float64
end

function PlaneStrain(prop::Dict{String, Any})
    E = prop["E"]; ν = prop["ν"]
    H = zeros(3,3)
    H[0,0] = E*(1. -ν)/((1+ν)*(1. -2. *ν));
    H[0,1] = H[0,0]*ν/(1-ν);
    H[1,0] = H[0,1];
    H[1,1] = H[0,0];
    H[2,2] = H[0,0]*0.5*(1. -2. *ν)/(1. -ν);
    PlaneStrain(H, E, ν)
end

function getStress(self::PlaneStrain, strain::Array{Float64})
    sigma = self.H * strain

    return sigma, self.H
end

function getTangent(self::PlaneStrain)
    self.H
end
