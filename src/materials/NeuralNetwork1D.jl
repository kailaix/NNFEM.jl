export  NeuralNetwork1D, getStress

mutable struct NeuralNetwork1D
    ρ::Float64 # density
    # hardening parameter, yield function = f - (σY + Kα)
    σ0::Float64 # stress at last time step
    σ0_::Float64 # σ0 to be updated in `commitHistory`
    ε0::Float64
    ε0_::Float64
    nn::Function
end


function NeuralNetwork1D(prop::Dict{String, Any})
    ρ = prop["rho"];
    nn = prop["nn"]
    σ0 = 0.0; σ0_ = 0.0; ε0 = 0.0; ε0_ = 0.0
    NeuralNetwork1D(ρ, σ0, σ0_, ε0, ε0_, nn)
end

function getStress(self::NeuralNetwork1D,  strain::Float64,  Dstrain::Float64, Δt::Float64 = 0.0)
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
    self.ε0 = self.ε0_
end
