using LinearAlgebra

function f(σ, α, σY, K)
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


ε = [100.0;200.0;1.0]
ε0 = [0.0;0.0;0.0]
σ0 = [0.0;0.0;0.0]
α0 = 0.0
H = UniformScaling(1.0)
σY = 1
K = 1


function compute(ipt)
    σ = ipt[1:3]; Δγ = ipt[4]
    α = α0 + Δγ
    r1 = σ + Δγ * H* fσ(σ) - σ0 - H*ε - H*ε0 
    r2 = f(σ, α, σY, K)
    J = [UniformScaling(1.0)+Δγ*H*fσσ(σ) H*fσ(σ)
        reshape(fσ(σ),1,3) -K]
    @show [r1;r2]
    return [r1;r2], J
end

function compute_all(ipt)
    σ = ipt[1:3]; Δγ = ipt[4]
    α = α0 + Δγ
    r1 = σ + Δγ * H* fσ(σ) - σ0 - H*ε - H*ε0 
    r2 = f(σ, α, σY, K)
        
    J = [UniformScaling(1.0)+Δγ*H*fσσ(σ) H*fσ(σ)
        reshape(fσ(σ),1,3) -K]
    δ = J\[H;zeros(1,3)]
    return δ[1:3,:]
end

function f1(σ)
    fσ(σ), fσσ(σ)
end
function g1(x)
    [sum(x^10); sum(x); sum(x)], [reshape(10x^9,1,3);ones(2,3)]
end
# compute(ones(4))
gradtest(compute, 100ones(4))
# gradtest(f1, rand(3))

