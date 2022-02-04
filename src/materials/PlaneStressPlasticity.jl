export  PlaneStressPlasticity, getStress

mutable struct PlaneStressPlasticity
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



function f(σ::Array{Float64}, α::Float64, σY::Float64, K::Float64)
    return sqrt(σ[1]^2-σ[1]*σ[2]+σ[2]^2+3*σ[3]^2)-σY-K*α
end

function fσ(σ::Array{Float64})
    σ1, σ2, σ3 = σ[1], σ[2], σ[3]
    J2 = sqrt(σ1^2-σ1*σ2+σ2^2+3*σ3^2)
    z = [(σ1 - σ2/2)/J2;
        (-σ1/2 + σ2)/J2;
        3*σ3/J2]
end

function fσσ(σ::Array{Float64})
    σ1, σ2, σ3 = σ[1], σ[2], σ[3]
    J2 = sqrt(σ1^2-σ1*σ2+σ2^2+3*σ3^2)
    [     (-σ1 + σ2/2)*(σ1 - σ2/2)/J2^3 + 1/J2 (σ1/2 - σ2)*(σ1 - σ2/2)/J2^3 - 1/(2*J2)                                   -3*σ3*(σ1 - σ2/2)/J2^3;
    (-σ1 + σ2/2)*(-σ1/2 + σ2)/J2^3 - 1/(2*J2)    (-σ1/2 + σ2)*(σ1/2 - σ2)/J2^3 + 1/J2                                  -3*σ3*(-σ1/2 + σ2)/J2^3;
    3*σ3*(-σ1 + σ2/2)/J2^3                                                        3*σ3*(σ1/2 - σ2)/J2^3 -9*σ3^2/J2^3 + 3/J2]
end

function PlaneStressPlasticity(prop::Dict{String, Any})
    E = prop["E"]; ν = prop["nu"]; ρ = prop["rho"];
    K = prop["K"]; σY = prop["sigmaY"]
    H = zeros(3,3)
    H[1,1] = E/(1. -ν*ν)
    H[1,2] = H[1,1]*ν
    H[2,1] = H[1,2]
    H[2,2] = H[1,1]
    H[3,3] = E/(2.0*(1.0+ν))
    σ0 = zeros(3); σ0_ = zeros(3); ε0 = zeros(3); ε0_ = zeros(3)
    PlaneStressPlasticity(H, E, ν, ρ, K, σY, 0.0, 0.0, σ0, σ0_, ε0, ε0_)
end

function getStress(self::PlaneStressPlasticity,  strain::Array{Float64},  Dstrain::Array{Float64}, Δt::Float64 = 0.0)

    Newton_maxiter = 10
    Newton_Abs_Err = 1e-8 
    Newton_Rel_Err = 1e-5


    # #@show "***", strain, Dstrain
    local dΔσdΔε
    ε = strain 
    ε0 = Dstrain 
    σ0 = self.σ0 
    α0 = self.α
    H = self.H 
    K = self.K 
    σY = self.σY 
    Δγ = 0.0
    σ = σ0 + H*(ε-ε0) 
    # σ = σ0
    α = α0 + Δγ

    r2 = f(σ, α, σY, K)
    #@show r2, α, K, σY
    if r2<0
        #@show "Elasticity"
        σ = σ0 + H*(ε-ε0) 
        dΔσdΔε = H
        #@show "elastic", α, α0, σ, σ0,ε, ε0, σ0 + H*(ε-ε0) , H*ε
    else
        # error()
        σ = σ0 + H*(ε-ε0) 
        function compute(σ, Δγ)
            α = α0 + Δγ
            r1 = σ + Δγ * H* fσ(σ) - σ0 - H*ε + H*ε0 
            r2 = f(σ, α, σY, K)
            J = [UniformScaling(1.0)+Δγ*H*fσσ(σ) H*fσ(σ)
                reshape(fσ(σ),1,3) -K]
            # #@show [r1;r2]
            return [r1;r2], J
        end

        function compute_sensitivity(σ, Δγ)
            α = α0 + Δγ
            J = [UniformScaling(1.0)+Δγ*H*fσσ(σ) H*fσ(σ)
                reshape(fσ(σ),1,3) -K]
            δ = J\[H;zeros(1,3)]
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
            if i==Newton_maxiter
                function f(∂∂u)
                    res, J = compute(∂∂u[1:3],∂∂u[4])
                end
                gradtest(f, [σ;Δγ])
                error("Newton in GetStress does not converge")
            end
        end

        # if f(σ, α, σY, K)>0
        #     @show sqrt(σ[1]^2-σ[1]*σ[2]+σ[2]^2+3*σ[3]^2), α, K
        # end

        dΔσdΔε = compute_sensitivity(σ, Δγ)
        # @show "plastic", α, α0, Δγ, σ,σ0,ε, ε0, σ0 + H*(ε-ε0) , H*ε
        # if σ[1]<0.0
        #     error()
        # end
        
    end
    # #@show Δγ
    self.α_ = self.α + Δγ
    self.σ0_ = σ[:]
    self.ε0_ = ε
    # self.σ0_ = self.σ0
    # #@show σ, dΔσdΔε
    return σ, dΔσdΔε
end

function getTangent(self::PlaneStressPlasticity)
    error("Not implemented")
end

function commitHistory(self::PlaneStressPlasticity)
    self.α = self.α_
    self.σ0 = self.σ0_
    self.ε0 = self.ε0_
    # if self.α>1e-15
    #     error("MATERIAL ERROR: Plasticity")
    # end
end
