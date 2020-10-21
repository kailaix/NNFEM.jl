using Revise
using NNFEM 
using PyPlot
using LinearAlgebra
using ADCME
using ADCMEKit
using DelimitedFiles 
using ProgressMeter
using PyCall
using MAT
using JLD2

NT = 50
Δt = 2.0/NT 

FILE = splitdir(pathof(NNFEM))[1]*"/../deps/Data/"
node, elem = meshread(FILE*"twoholes.msh")
node *= 100
xmax = maximum(node[:,1])
ymax = maximum(node[:,2])

elements = []
prop = Dict("name"=> "PlaneStressIncompressibleRivlinSaunders", "rho"=> 1.522,  "C1"=>0.162, "C2"=>5.9e-3)

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
        if elem.coords[k,2]<1e-5 && elem.coords[k+1>4 ? 1 : k+1,2]<1e-5
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
    if node[i,1]>xmax-1e-5 || node[i,2]>ymax-1e-5
        EBC[i,:] .= -1
    end
end

domain = Domain(node, elements, 2, EBC, g, FBC, f, Edge_Traction_Data)

# Set initial condition: at reset
Dstate = zeros(domain.neqs) 
state = zeros(domain.neqs)
velo = zeros(domain.neqs)
acce = zeros(domain.neqs)
EBC_func = nothing 
FBC_func = nothing 
Body_func = nothing 
# Construct Edge_func
function Edge_func_hyperelasticity(x, y, t, idx)
  return [zeros(length(x)) 0.01*ones(length(x)) * sin(2π*t)] 
end

globaldata = GlobalData(state, Dstate, velo, acce, domain.neqs, EBC_func, FBC_func,Body_func, Edge_func_hyperelasticity)


assembleMassMatrix!(globaldata, domain)
SolverInitial!(Δt, globaldata, domain)


updateStates!(domain, globaldata)
ω = EigenMode(Δt, globaldata, domain)
@show "stable time step is ", 0.8 * 2/ω, " current time step is ", Δt
@showprogress for i = 1:NT
    global globaldata, domain = GeneralizedAlphaSolverStep(globaldata, domain, Δt)
end

p = visualize_von_mises_stress(domain)
saveanim(p, raw"C:\Users\kaila\Desktop\ADCMEImages\NNFEM\visualize_von_mises_stress.gif")


visualize_von_mises_stress(domain, 10)
savefig(raw"C:\Users\kaila\Desktop\ADCMEImages\NNFEM\visualize_von_mises_stress_50.png")

p = visualize_displacement(domain, scale_factor = 20.0)
saveanim(p, raw"C:\Users\kaila\Desktop\ADCMEImages\NNFEM\visualize_displacement.gif")


visualize_mesh(domain.nodes, getElems(domain))
visualize_mesh(domain)
savefig(raw"C:\Users\kaila\Desktop\ADCMEImages\NNFEM\visualize_mesh.png")

visualize_boundary(domain)
savefig(raw"C:\Users\kaila\Desktop\ADCMEImages\NNFEM\visualize_boundary.png")

d_ = hcat(domain.history["state"]...)'|>Array
visualize_total_deformation_on_scoped_body(d_, domain; scale_factor=20)
saveanim(p, raw"C:\Users\kaila\Desktop\ADCMEImages\NNFEM\visualize_total_deformation_on_scoped_body.gif")


visualize_von_mises_stress_on_scoped_body(d_, domain; scale_factor=20)
saveanim(p, raw"C:\Users\kaila\Desktop\ADCMEImages\NNFEM\visualize_von_mises_stress_on_scoped_body.gif")
