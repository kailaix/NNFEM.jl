using Revise
using ADCME
using NNFEM
using JLD2
using PyCall
using LinearAlgebra
using Random, Distributions


reset_default_graph()

E = 200e3


nntype = "ae_scaled"

#nntype = "piecewise"

idx = 1
H_function = "None"
include("nnutil.jl")

n = 50
X_1d = LinRange(-2.0, 2.0, n)
X = zeros(Float64, n, 1)
for i = 1:n
    X[i, 1] = X_1d[i]
end

ε_μ = 0.0; ε_σ = 0.00
noise = rand(Normal(ε_μ, ε_σ), (n,1))
Y = X * E + noise



loss = constant(0.0)
x = constant(X)
dummy = x

#ε = x; ε0 = x; σ0 = x


y = nn(x, dummy, dummy)



loss += mean((y-Y)^2) 



sess = Session(); init(sess)
@show run(sess, loss)
for i = 1:10
    BFGS!(sess, loss, 1000)
    ADCME.save(sess, "Data/NNLearn_$(nntype)_ite$(i).mat")
end
