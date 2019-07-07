export  Plasticity1D, getStress

mutable struct Plasticity1D
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


function Plasticity1D(prop::Dict{String, Any})
    E = prop["E"]; ν = prop["nu"]; ρ = prop["rho"];
    K = prop["K"]; σY = prop["sigmaY"]
    σ0 = 0.0; σ0_ = 0.0
    Plasticity1D(H, E, ν, ρ, K, σY, 0.0, 0.0, σ0, σ0_)
end

function getStress(self::PlaneStressPlasticity,  strain::Array{Float64},  Dstrain::Array{Float64})
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
    if r2<0
        #@show "Elasticity"
        σ = σ0 + H*(ε-ε0) 
        dΔσdΔε = H
        #@show "elastic", α, α0, σ, σ0,ε, ε0, σ0 + H*(ε-ε0) , H*ε
    else
        # @show "Plasticity"
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
        for i = 1:10
            res, J = compute(σ, Δγ)
            δ = -J\res
            σ += δ[1:3]; Δγ += δ[4]
            # #@show norm(res)/norm(res0)
            if norm(res)/norm(res0)<1e-5
                break
            end
            if i==100
                function f(∂∂u)
                    res, J = compute(∂∂u[1:3],∂∂u[4])
                end
                gradtest(f, [σ;Δγ])
                error("Newton in GetStress does not converge")
            end
        end

        dΔσdΔε = compute_sensitivity(σ, Δγ)
        #@show "plastic", α, α0, Δγ, σ,σ0,ε, ε0, σ0 + H*(ε-ε0) , H*ε
        # if σ[1]<0.0
        #     error()
        # end
        
    end
    # #@show Δγ
    self.α_ = self.α + Δγ
    self.σ0_ = σ[:]
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
end
