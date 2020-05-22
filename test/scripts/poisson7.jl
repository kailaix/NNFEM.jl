#=
process data4.mat 
vectorize the neural network weights and biases
=#
using Revise
using NNFEM
using PoreFlow
using ADCME 
using LinearAlgebra
using PyPlot
using MATLAB
using PyCall
using MAT
include("common1.jl")

d = matread("data/4.mat")
n_layers = length(d)÷2
ks = collect(keys(d))
θ = Array{Float64}[]
k = filter(x->occursin("connectedbackslashweights", x), ks)[1]
push!(θ, d[k]'[:])
k = filter(x->occursin("connectedbackslashbiases", x), ks)[1]
if typeof(d[k])<:Real
    d[k] = [d[k]]
end
push!(θ, d[k][:])
for i = 1:n_layers-1
    k = filter(x->occursin("connected_$(i)backslashweights", x), ks)[1]
    push!(θ, d[k]'[:])
    k = filter(x->occursin("connected_$(i)backslashbiases", x), ks)[1]
    if typeof(d[k])<:Real
        d[k] = [d[k]]
    end
    push!(θ, d[k][:])
end 
θ = vcat(θ...)

matwrite("data/7.mat", Dict(
    "theta"=>θ
))