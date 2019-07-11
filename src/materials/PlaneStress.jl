export  PlaneStress
mutable struct PlaneStress
    H::Array{Float64}
    E::Float64
    ν::Float64
    ρ::Float64
end

function PlaneStress(prop::Dict{String, Any})
    E = prop["E"]; ν = prop["nu"]; ρ = prop["rho"]
    H = zeros(3,3)

    H[1,1] = E/(1. -ν*ν)
    H[1,2] = H[1,1]*ν
    H[2,1] = H[1,2]
    H[2,2] = H[1,1]
    H[3,3] = E/(2.0*(1.0+ν))
    @show H
    PlaneStress(H, E, ν, ρ)
end

function getStress(self::PlaneStress, strain::Array{Float64}, Dstrain::Array{Float64}, Δt::Float64 = 0.0)
    sigma = self.H * strain

    return sigma, self.H
end

function getTangent(self::PlaneStress)
    self.H
end

function commitHistory(self::PlaneStress)
end
