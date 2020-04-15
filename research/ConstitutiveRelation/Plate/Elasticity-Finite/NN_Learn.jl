using Revise
using ADCME
using NNFEM
using JLD2
using PyCall
using LinearAlgebra
using Random, Distributions


reset_default_graph()

#ρ = 1153.4 kg/m^3 = 1153.4 kg/m^3 * 0.000076 m = 0.0876584 kg/m^2
#E = 944836855.0kg/m/s^2 * 0.000076m = 71807.60098kg/s^2 = 0.07180760098kg/ms^2
prop = Dict("name"=> "PlaneStress", "rho"=> 0.0876584, "E"=>0.07180760098, "nu"=>0.4)

stress_scale = 1.0
strain_scale = 1.0

mat = PlaneStress(prop)
H0 = mat.H

nntype = "ae_scaled"
idx = 1
H_function = "None"
include("nnutil.jl")


n = 50
X_1d = LinRange(-2.0, 2.0, n)
X = zeros(Float64,  n^3, 3)
for i1 = 1:n
    for i2 = 1:n
        for i3 = 1:n
            X[(i1-1)*n^2 + (i2-1)*n + i3, :] .= X_1d[i1], X_1d[i2], X_1d[i3]
        end
    end
end
ε_μ = 0.0; ε_σ = 0.00
noise = rand(Normal(ε_μ, ε_σ), (n^3,3))
Y = X * H0 + noise


loss = constant(0.0)
x = constant(X)
dummy = x
y = nn(x[:,1:3], dummy, dummy)
loss += mean((y-Y)^2) 



sess = Session(); init(sess)
@show run(sess, loss)
for i = 1:10
    BFGS!(sess, loss, 1000)
    ADCME.save(sess, "Data/NNLearn_$(nntype)_ite$(i).mat")
end
