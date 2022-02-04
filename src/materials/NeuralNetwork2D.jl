export  NeuralNetwork2D, getStress

mutable struct NeuralNetwork2D
    ρ::Float64 # density
    # hardening parameter, yield function = f - (σY + Kα)
    σ0::Array{Float64} # stress at last time step
    σ0_::Array{Float64} # σ0 to be updated in `commitHistory`
    ε0::Array{Float64} 
    ε0_::Array{Float64} 
    α0::Union{Missing, Array{Float64}}
    α0_::Union{Missing, Array{Float64}}
    nn::Function
end


function NeuralNetwork2D(prop::Dict{String, Any})
    ρ = prop["rho"];
    nn = prop["nn"]
    α0 = missing; α0_ = missing
    if haskey(prop, "n_internal")
        n_internal = prop["n_internal"]
        @assert isa(n_internal, Integer) && n_internal>0
        α0 = zeros(n_internal); α0_ = zeros(n_internal)
        # @show α0
    end
    σ0 = zeros(3); σ0_ = zeros(3); ε0 = zeros(3); ε0_ = zeros(3); 
    NeuralNetwork2D(ρ, σ0, σ0_,ε0,ε0_, α0, α0_, nn)
end

function getStress(self::NeuralNetwork2D,  strain::Array{Float64},  Dstrain::Array{Float64}, Δt::Float64 = 0.0)
    # #@show "***", strain, Dstrain
    ε = strain 
    ε0 = Dstrain 
    σ0 = self.σ0 
    if ismissing(self.α0_)
        σ, dΔσdΔε = self.nn(ε, ε0, σ0, Δt)
        self.σ0_ = σ
        self.ε0_ = ε
        return σ, dΔσdΔε
    else
        # @show self.α0
        σ, dΔσdΔε, α = self.nn(ε, ε0, σ0, Δt, self.α0)
        self.σ0_ = σ
        self.ε0_ = ε
        self.α0_ = α
        return σ, dΔσdΔε
    end
end

function getTangent(self::NeuralNetwork2D)
    error("Not implemented")
end

function commitHistory(self::NeuralNetwork2D)
    self.σ0 = self.σ0_
    self.ε0 = self.ε0_
    self.α0 = self.α0_
end
