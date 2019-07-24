export  NeuralNetwork1D, getStress

mutable struct NeuralNetwork1D
    ρ::Float64 # density
    # hardening parameter, yield function = f - (σY + Kα)
    σ0::Array{Float64} # stress at last time step
    σ0_::Array{Float64} # σ0 to be updated in `commitHistory`
    nn::Function
end


function NeuralNetwork1D(prop::Dict{String, Any})
    ρ = prop["rho"];
    nn = prop["nn"]
    σ0 = zeros(1); σ0_ = zeros(1)
    NeuralNetwork1D(ρ, σ0, σ0_, nn)
end

function getStress(self::NeuralNetwork1D,  strain::Array{Float64},  Dstrain::Array{Float64}, Δt::Float64 = 0.0)
    # #@show "***", strain, Dstrain
    ε = strain 
    ε0 = Dstrain 
    σ0 = self.σ0 
    σ, dΔσdΔε = self.nn(ε, ε0, σ0, Δt)
    self.σ0_ = σ
    return σ, dΔσdΔε
end

function getTangent(self::NeuralNetwork1D)
    error("Not implemented")
end

function commitHistory(self::NeuralNetwork1D)
    self.σ0 = self.σ0_
end
