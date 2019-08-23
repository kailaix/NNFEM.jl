export  PlaneStress
mutable struct PlaneStress
    H::Array{Float64}
    E::Float64
    ν::Float64
    ρ::Float64
    σ0::Array{Float64} # stress at last time step
    σ0_::Array{Float64} # σ0 to be updated in `commitHistory`
    ε0::Array{Float64} 
    ε0_::Array{Float64} 
end

function PlaneStress(prop::Dict{String, Any})
    E = prop["E"]; ν = prop["nu"]; ρ = prop["rho"]
    H = zeros(3,3)

    H[1,1] = E/(1. -ν*ν)
    H[1,2] = H[1,1]*ν
    H[2,1] = H[1,2]
    H[2,2] = H[1,1]
    H[3,3] = E/(2.0*(1.0+ν))
    # @show H
    # # ! remove it!!!
    # H = [ 4.09627e12  2.88053e11  -1.77231   
    #     2.88053e11  8.29483e11   0.616748  
    # -1.77231     0.616748     3.43662e11]
    σ0 = zeros(3); σ0_ = zeros(3); ε0 = zeros(3); ε0_ = zeros(3)
    PlaneStress(H, E, ν, ρ, σ0, σ0_,ε0,ε0_)
end

function getStress(self::PlaneStress, strain::Array{Float64}, Dstrain::Array{Float64}, Δt::Float64 = 0.0)
    sigma = self.H * strain
    self.σ0_ = copy(sigma)
    self.ε0_ = copy(strain)
    return sigma, self.H
end

function getTangent(self::PlaneStress)
    self.H
end

function commitHistory(self::PlaneStress)
    self.σ0 = self.σ0_
    self.ε0 = self.ε0_
end
