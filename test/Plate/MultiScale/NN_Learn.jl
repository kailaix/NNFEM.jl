include("nnutil.jl")

# H0 = constant(H1/stress_scale)
testtype = "NeuralNetwork2D"
force_scale = 6.0
nntype = "piecewise"

# ! define H0
# Trained with nx, ny = 10, 5
H0 = [1.26827e6       3.45169e5   -5187.35
      3.45169e5       1.25272e6  -10791.7
      -5187.35       -10791.7        536315.0]


n_data = [100, 201, 202, 203]

strain = Array{Array{Float64}}(undef, length(n_data))
stress = Array{Array{Float64}}(undef, length(n_data))
for (k,tid) in enumerate(n_data)
    @load "Data/order2/domain$(tid)_6.0_5.jld2" domain
    strain[k] = vcat(domain.history["strain"]...) 
    stress[k] = vcat(domain.history["stress"]...) 
end




