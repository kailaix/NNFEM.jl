export  PathDependent2D, getStress

mutable struct PathDependent2D
    H::Array{Float64} # tangent matrix for plane stress
    E::Float64 # Young's modulus
    ν::Float64 # Poisson's ratio
    ρ::Float64 # density
    σ0::Array{Float64} # stress at last time step
    σ0_::Array{Float64} # σ0 to be updated in `commitHistory`
    ε0::Array{Float64} # stress at last time step
    ε0_::Array{Float64} # σ0 to be updated in `commitHistory`
end


@doc """
    The power-law hardening law has the the following format
    σY = (0.1 + 0.3 ε_equiv^0.4)MPa

""" -> 
function f_σY(σ, σY, K)
    return sqrt(σ[1]^2-σ[1]*σ[2]+σ[2]^2+3*σ[3]^2)-σY-K*α
end

function fσ(σ)
    σ1, σ2, σ3 = σ[1], σ[2], σ[3]
    J2 = sqrt(σ1^2-σ1*σ2+σ2^2+3*σ3^2)
    z = [(σ1 - σ2/2)/J2;
        (-σ1/2 + σ2)/J2;
        3*σ3/J2]
end

function fσσ(σ)
    σ1, σ2, σ3 = σ[1], σ[2], σ[3]
    J2 = sqrt(σ1^2-σ1*σ2+σ2^2+3*σ3^2)
    [     (-σ1 + σ2/2)*(σ1 - σ2/2)/J2^3 + 1/J2 (σ1/2 - σ2)*(σ1 - σ2/2)/J2^3 - 1/(2*J2)                                   -3*σ3*(σ1 - σ2/2)/J2^3;
    (-σ1 + σ2/2)*(-σ1/2 + σ2)/J2^3 - 1/(2*J2)    (-σ1/2 + σ2)*(σ1/2 - σ2)/J2^3 + 1/J2                                  -3*σ3*(-σ1/2 + σ2)/J2^3;
    3*σ3*(-σ1 + σ2/2)/J2^3                                                        3*σ3*(σ1/2 - σ2)/J2^3 -9*σ3^2/J2^3 + 3/J2]
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
    H = self.H
    
    σ = σ0 + H*(ε-ε0) 
    σY = f_σY(ε)
    #@show r2, α, K, σY
    if σ < σY
        #@show "Elasticity"
        σ = σ0 + H*(ε-ε0) 
        dΔσdΔε = H
        #@show "elastic", α, α0, σ, σ0,ε, ε0, σ0 + H*(ε-ε0) , H*ε
    else
        # error()
        σ = σY
        dΔσdΔε = df_σY(ε)
        
    end
    # #@show Δγ
    self.σ0_ = σ[:]
    self.ε0_ = ε
    # self.σ0_ = self.σ0
    # #@show σ, dΔσdΔε
    return σ, dΔσdΔε
end

function getTangent(self::PathDependent2D)
    error("Not implemented")
end

function commitHistory(self::PathDependent2D)
    self.σ0 = self.σ0_
    self.ε0 = self.ε0_
end
