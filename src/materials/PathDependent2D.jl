export  PathDependent2D, getStress

mutable struct PathDependent2D
    H::Array{Float64} # tangent matrix for plane stress
    E::Float64 # Young's modulus
    ν::Float64 # Poisson's ratio
    ρ::Float64 # density
    # hardening parameter, yield function = f - (σY + Kα)
    K::Float64 
    σY::Float64
    α::Float64 
    α_::Float64 # α to be updated in `commitHistory`
    σ0::Array{Float64} # stress at last time step
    σ0_::Array{Float64} # σ0 to be updated in `commitHistory`
    ε0::Array{Float64} # stress at last time step
    ε0_::Array{Float64} # σ0 to be updated in `commitHistory`
end


function PathDependent2D(prop::Dict{String, Any})
    E = prop["E"]; ν = prop["nu"]; ρ = prop["rho"];
    K = prop["K"]; σY = prop["sigmaY"]
    H = zeros(3,3)
    H[1,1] = E/(1. -ν*ν)
    H[1,2] = H[1,1]*ν
    H[2,1] = H[1,2]
    H[2,2] = H[1,1]
    H[3,3] = E/(2.0*(1.0+ν))
    σ0 = zeros(3); σ0_ = zeros(3); ε0 = zeros(3); ε0_ = zeros(3)
    PathDependent2D(H, E, ν, ρ, K, σY, 0.0, 0.0, σ0, σ0_, ε0, ε0_)
end



@doc """
    For debugging pathdependent materials, we an build arbitrary pathdependent material law

    sigma = sigma0 + (eps - eps0)**2

""" -> 
function getStress(self::PathDependent2D,  strain::Array{Float64},  Dstrain::Array{Float64}, Δt::Float64 = 0.0)
    
    local dΔσdΔε
    ε = strain 
    ε0 = Dstrain 
    σ0 = self.σ0 
    
    E = self.E
    
    α = 100
    σ = σ0 + E*(ε-ε0) #* (1.0 + α*ε[1] + ε[2] + ε[3])
    dΔσdΔε  =  [E 0 0 ; 0 E 0; 0 0 E] #* (1.0 + α*ε[1] + ε[2] + ε[3]) + [α*H*(ε-ε0) H*(ε-ε0) H*(ε-ε0)]

    #@show ε[1] , ε[2] , ε[3]

    self.σ0_ = σ
    self.ε0_ = ε

    return σ, dΔσdΔε
end

function getTangent(self::PathDependent2D)
    error("Not implemented")
end

function commitHistory(self::PathDependent2D)
    self.σ0 = self.σ0_
    self.ε0 = self.ε0_
end
