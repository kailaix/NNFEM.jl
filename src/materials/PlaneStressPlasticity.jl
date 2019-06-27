export  PlaneStressPlasticity

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
end


function f(σ, α, σY, K)
    return sqrt(σ[1]^2-σ[1]*σ[2]+σ[2]^2+3*σ[3]^2)-σY-K*α
end

function fσ(σ)
    σ1, σ2, σ3 = σ[1], σ[2], σ[3]
    v = sqrt(σ1^2-σ1*σ2+σ2^2+3*σ3^2)
    z = [(σ1 - σ2/2)/sqrt(σ1^2 - σ1*σ2 + σ2^2 + 3*σ3^2);
        (-σ1/2 + σ2)/sqrt(σ1^2 - σ1*σ2 + σ2^2 + 3*σ3^2);
        3*σ3/sqrt(σ1^2 - σ1*σ2 + σ2^2 + 3*σ3^2)]
end

function fσσ(σ)
    σ1, σ2, σ3 = σ[1], σ[2], σ[3]
    [     (-σ1 + σ2/2)*(σ1 - σ2/2)/(σ1^2 - σ1*σ2 + σ2^2 + 3*σ3^2)^(3/2) + 1/sqrt(σ1^2 - σ1*σ2 + σ2^2 + 3*σ3^2) (σ1/2 - σ2)*(σ1 - σ2/2)/(σ1^2 - σ1*σ2 + σ2^2 + 3*σ3^2)^(3/2) - 1/(2*sqrt(σ1^2 - σ1*σ2 + σ2^2 + 3*σ3^2))                                   -3*σ3*(σ1 - σ2/2)/(σ1^2 - σ1*σ2 + σ2^2 + 3*σ3^2)^(3/2);
(-σ1 + σ2/2)*(-σ1/2 + σ2)/(σ1^2 - σ1*σ2 + σ2^2 + 3*σ3^2)^(3/2) - 1/(2*sqrt(σ1^2 - σ1*σ2 + σ2^2 + 3*σ3^2))    (-σ1/2 + σ2)*(σ1/2 - σ2)/(σ1^2 - σ1*σ2 + σ2^2 + 3*σ3^2)^(3/2) + 1/sqrt(σ1^2 - σ1*σ2 + σ2^2 + 3*σ3^2)                                  -3*σ3*(-σ1/2 + σ2)/(σ1^2 - σ1*σ2 + σ2^2 + 3*σ3^2)^(3/2);
3*σ3*(-σ1 + σ2/2)/(σ1^2 - σ1*σ2 + σ2^2 + 3*σ3^2)^(3/2)                                                        3*σ3*(σ1/2 - σ2)/(σ1^2 - σ1*σ2 + σ2^2 + 3*σ3^2)^(3/2) -9*σ3^2/(σ1^2 - σ1*σ2 + σ2^2 + 3*σ3^2)^(3/2) + 3/sqrt(σ1^2 - σ1*σ2 + σ2^2 + 3*σ3^2)]
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
    σ0 = zeros(3); σ0_ = zeros(3)
    PlaneStressPlasticity(H, E, ν, ρ, K, σY, 0.0, 0.0, σ0, σ0_)
end

function getStress(self::PlaneStressPlasticity,  strain::Array{Float64},  Dstrain::Array{Float64})
    local res0
    σA = self.σ0; H = self.H; Δε = strain - Dstrain
    σtrial = σA + H*Δε
    Δσ = zeros(3); Δγ = 0.0
    dΔσdΔε = zeros(3,3)
    # http://homes.civil.aau.dk/lda/continuum/plast.pdf
    if f(σtrial, self.α, self.σY, self.K)<=0 
        Δσ = H*Δε
        Δγ = 0.0
        dΔσdΔε = H
    else
        # error("Not possible")
        # Newton Raphson iteration
        # Equation 1: Δσ - HΔε + HΔλ∇f(σA+Δσ) = 0
        # Equation 2: f(σA + Δσ) = 0
        # (Δσ, Δλ)
        Δσtrial = H*Δε
        for iter = 1:100
            σtrial = σA + Δσtrial
            q1 = Δσtrial - H*Δε + Δγ*H*fσ(σtrial)
            q2 = f(σtrial, self.α, self.σY, self.K)

            # * check error 
            res = sum(q1.^2)+q2 
            if iter==1
                res0 = res
            end
            if iter==1000
                error("Newton iteration fails, err=$(e/e0), input norm=$(norm(Δε[3(i-1)+1:3i]))")
            end
            if res/res0<1e-8
                # @info "$iter, $q2", Δσtrial, H*Δε
                break
            end
            J = [UniformScaling(1.0)+Δγ*H*fσσ(σtrial) H*fσ(σtrial);
                    reshape(fσ(σtrial),1,3) 0]
            δ = -J\[q1;q2]
            Δσtrial += δ[1:3]
            Δγ += δ[4]
        end
        Δσ = Δσtrial
        σ = σA + Δσ
        A = [UniformScaling(1.0)+Δγ*H*fσσ(σ) H*fσ(σ);
                reshape(fσ(σ),1,3) 0]
        rhs = [H;zeros(1,3)]
        out = A\rhs
        dΔσdΔε = out[1:3,:]
    end
    σ = σA + Δσ
    # q1 = Δσ - H*Δε + Δγ*H*fσ(σ)
    # q2 = f(σ, self.α, self.σY, self.K)
    # @show norm(q1), q2
    # @info Δσ, H*Δε, norm(Δσ-H*Δε)≈0

    # @info "*",self.σ0, self.σ0_, σ,  H*strain, H*Dstrain
    
    self.α_ = self.α + Δγ
    # self.σ0_ = σ[:]
    
    return σ, dΔσdΔε
end

function getTangent(self::PlaneStressPlasticity)
    error("Not implemented")
end

function commitHistory(self::PlaneStressPlasticity)
    self.α = self.α_
    # self.σ0 = self.σ0_
end
