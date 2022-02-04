export  PlaneStressPlasticityLawBased, getStress

mutable struct PlaneStressPlasticityLawBased
    H::Array{Float64} # tangent matrix for plane stress
    E::Float64 # Young's modulus
    ν::Float64 # Poisson's ratio
    ρ::Float64 # density
    a::Array{Float64}  # explicit hardening law parameters
    σ0::Array{Float64} # stress at last time step
    σ0_::Array{Float64} # σ0 to be updated in `commitHistory`
    ε0::Array{Float64} # stress at last time step
    ε0_::Array{Float64} # σ0 to be updated in `commitHistory`
end


@doc """
    The yield function 
        \bar{σ} - σY(\bar{\epsilon} ,  a)

    σY is described by a hardening law, for example
    The power-law:
            σY = a[1] + a[2] ε_equiv^a[3]


    To modify the law, you need to modify both function f(σ, ε, a) and fε(ε, a)

""" -> 
function f(σ, ε, a)
    # von Mises equivalent strain
    ε_equiv = sqrt(4.0/9.0*(ε[1]^2 + ε[2]^2 - ε[1]*ε[2]) + 1.0/3.0*ε[3]^2)
    σY = a[1] + a[2]*ε_equiv^a[3]
    return sqrt(σ[1]^2-σ[1]*σ[2]+σ[2]^2+3*σ[3]^2)-σY
end

function fσ(σ)
    σ1, σ2, σ3 = σ[1], σ[2], σ[3]
    J2 = sqrt(σ1^2-σ1*σ2+σ2^2+3*σ3^2)
    z = [(σ1 - σ2/2)/J2;
        (-σ1/2 + σ2)/J2;
        3*σ3/J2]
end

function fε(ε, a)
    ε1, ε2, ε3 = ε[1], ε[2], ε[3]
    ε_equiv = sqrt(4.0/9.0*(ε[1]^2 + ε[2]^2 - ε[1]*ε[2]) + 1.0/3.0*ε[3]^2)

    z = [(4.0/9.0 * ε1 - 2.0/9.0 * ε2)/ε_equiv;
         (4.0/9.0 * ε2 - 2.0/9.0 * ε1)/ε_equiv;
         1.0/3.0 * ε3/ε_equiv] * (a[2]*a[3]*ε_equiv^(a[3] - 1.0))

    return z
end

function fσσ(σ)
    σ1, σ2, σ3 = σ[1], σ[2], σ[3]
    J2 = sqrt(σ1^2-σ1*σ2+σ2^2+3*σ3^2)
    [     (-σ1 + σ2/2)*(σ1 - σ2/2)/J2^3 + 1/J2 (σ1/2 - σ2)*(σ1 - σ2/2)/J2^3 - 1/(2*J2)                                   -3*σ3*(σ1 - σ2/2)/J2^3;
    (-σ1 + σ2/2)*(-σ1/2 + σ2)/J2^3 - 1/(2*J2)    (-σ1/2 + σ2)*(σ1/2 - σ2)/J2^3 + 1/J2                                  -3*σ3*(-σ1/2 + σ2)/J2^3;
    3*σ3*(-σ1 + σ2/2)/J2^3                                                        3*σ3*(σ1/2 - σ2)/J2^3 -9*σ3^2/J2^3 + 3/J2]
end


function PlaneStressPlasticityLawBased(prop::Dict{String, Any})
    E = prop["E"]; ν = prop["nu"]; ρ = prop["rho"]; a = prop["args"]
    H = zeros(3,3)
    H[1,1] = E/(1. -ν*ν)
    H[1,2] = H[1,1]*ν
    H[2,1] = H[1,2]
    H[2,2] = H[1,1]
    H[3,3] = E/(2.0*(1.0+ν))
    σ0 = zeros(3); σ0_ = zeros(3); ε0 = zeros(3); ε0_ = zeros(3)
    PlaneStressPlasticityLawBased(H, E, ν, ρ, a, σ0, σ0_, ε0, ε0_)
end



@doc """
    For debugging pathdependent materials, we an build arbitrary pathdependent material law

    sigma = sigma0 + (eps - eps0)**2

""" -> 
function getStress(self::PlaneStressPlasticityLawBased,  strain::Array{Float64},  Dstrain::Array{Float64}, Δt::Float64 = 0.0)
    Newton_maxiter = 10
    Newton_Abs_Err = 1e-8 
    Newton_Rel_Err = 1e-5
    
    
    local dΔσdΔε
    ε = strain 
    ε0 = Dstrain 
    σ0 = self.σ0 
    
    E = self.E
    H = self.H
    a = self.a

    Δγ = 0.0
    
    σ = σ0 + H*(ε-ε0) 
    r2 = f(σ, ε, a)
    #@show r2, α, K, σY
    if r2 < 0
        #@show "Elasticity"
        σ = σ0 + H*(ε-ε0) 
        dΔσdΔε = H
        #@show "elastic", α, α0, σ, σ0,ε, ε0, σ0 + H*(ε-ε0) , H*ε
    else
        # solve for Δγ
        function compute(σ, Δγ)
            r1 = σ - (σ0 + H*(ε - ε0) - Δγ * H* fσ(σ))  
            r2 = f(σ, ε, a)
            J = [UniformScaling(1.0)+Δγ*H*fσσ(σ)  H*fσ(σ)
                 reshape(fσ(σ),1,3)                  0]
            return [r1;r2], J
        end

        function compute_sensitivity(σ, Δγ)
            J = [UniformScaling(1.0)+Δγ*H*fσσ(σ)  H*fσ(σ)
                 reshape(fσ(σ),1,3)                  0]
            
            δ = J\[H ;reshape(fε(ε, a),1,3)]

            return δ[1:3,:]
        end
        res0, _ = compute(σ, Δγ)
        for i = 1:Newton_maxiter
            res, J = compute(σ, Δγ)
            δ = -J\res
            σ += δ[1:3]; Δγ += δ[4]
            # #@show norm(res)/norm(res0)
            if norm(res)/norm(res0) < Newton_Rel_Err || norm(res) < Newton_Abs_Err
                break
            end
            if i == Newton_maxiter
                function f(∂∂u)
                    res, J = compute(∂∂u[1:3],∂∂u[4])
                end
                gradtest(f, [σ;Δγ])
                error("Newton in GetStress does not converge ", res, res0, norm(res)/norm(res0))
            end
        end

        # if f(σ, α, σY, K)>0
        #     @show sqrt(σ[1]^2-σ[1]*σ[2]+σ[2]^2+3*σ[3]^2), α, K
        # end

        dΔσdΔε = compute_sensitivity(σ, Δγ)
        
    end
    # #@show Δγ
    self.σ0_ = σ[:]
    self.ε0_ = ε
    # self.σ0_ = self.σ0
    #@show σ, dΔσdΔε


    return σ, dΔσdΔε
end

function getTangent(self::PlaneStressPlasticityLawBased)
    error("Not implemented")
end

function commitHistory(self::PlaneStressPlasticityLawBased)
    self.σ0 = self.σ0_
    self.ε0 = self.ε0_
end
