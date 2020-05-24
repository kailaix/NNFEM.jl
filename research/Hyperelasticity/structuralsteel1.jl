#=
Common data
=#
using Revise
using NNFEM 
using PyPlot
using LinearAlgebra
using ADCME
using ADCMEKit
using MAT 


NT = 1000
Δt = 0.1/NT 

node, elem = meshread("twoholes.msh")
node *= 10
xmax = maximum(node[:,1])

elements = []
prop = Dict("name"=> "PlaneStressIncompressibleRivlinSaunders", "rho"=> 7850,  "E"=>2e11, "nu"=>0.3,
        "C1"=>3.3017e5, "C2"=>30515.0)
for i = 1:size(elem,1)
    nodes = node[elem[i,:], :]
    elnodes = elem[i,:]
    ngp = 3
    push!(elements, FiniteStrainContinuum(nodes, elnodes, prop, ngp))
end

Edge_Traction_Data = Array{Int64}[]
for i = 1:length(elements)
    elem = elements[i]
    for k = 1:4
        if elem.coords[k,1]>xmax-1e-5 && elem.coords[k+1>4 ? 1 : k+1,1]>xmax-1e-5
            push!(Edge_Traction_Data, Int64[i, k, 1])
        end
    end
end
Edge_Traction_Data = hcat(Edge_Traction_Data...)'|>Array


EBC = zeros(Int64, size(node, 1), 2)
FBC = zeros(Int64, size(node, 1), 2)
g = zeros(size(node, 1), 2)
f = zeros(size(node, 1), 2)
for i = 1:size(node, 1)
    if node[i,1]<1e-5
        EBC[i,:] .= -1
    end
end

domain = Domain(node, elements, 2, EBC, g, FBC, f, Edge_Traction_Data)

Dstate = zeros(domain.neqs) 
state = zeros(domain.neqs)
velo = zeros(domain.neqs)
acce = zeros(domain.neqs)
EBC_func = nothing 
FBC_func = nothing 
Body_func = nothing 
# Construct Edge_func
function Edge_func(x, y, t, idx)
  return [zeros(length(x)) 1000*t*ones(length(x))]
end
globaldata = GlobalData(state, Dstate, velo, acce, domain.neqs, EBC_func, FBC_func,Body_func, Edge_func)

# assembleMassMatrix!(globaldata, domain)
# SolverInitial!(Δt, globaldata, domain)

# updateStates!(domain, globaldata)
# ω = EigenMode(Δt, globaldata, domain)
# @show "stable time step is ", 0.8 * 2/ω, " current time step is ", Δt
# for i = 1:NT
#     @info i 
#     global globaldata, domain = GeneralizedAlphaSolverStep(globaldata, domain, Δt)
#     # global globaldata, domain = ExplicitSolverStep(globaldata, domain, Δt)
# end
# d_ = hcat(domain.history["state"]...)'|>Array

# visualize_total_deformation_on_scoped_body(d_, domain;scale_factor=4.562)


function sample_interior(N, n)
    Random.seed!(233+n)
    s = Set([])
    while length(s)<n 
        i = rand(1:N)
        if i in s
            continue 
        else
            push!(s, i)
        end
    end
    Int64.(collect(s))
end


