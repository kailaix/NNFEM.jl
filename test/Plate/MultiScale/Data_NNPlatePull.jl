tid = parse(Int64, ARGS[1])
θ = 2π*(tid-1)/8
printstyled("tid=$tid\n", color=:green)

using Revise
using Test 
using NNFEM
using PyCall
using PyPlot
using JLD2
using ADCME
using LinearAlgebra

# tid = 1
"""
Property:
The matrix is made of titanium, which are assumed to be elasto-plastic titanium material,  
ρ = 4.5 g/cm^3;  E = 100GPa	 K=10e+9  ν =0.2   σY=970 MPa

Fibers are made of SiC, which are assumed to be isotropic and elastic, with
https://www.azom.com/properties.aspx?ArticleID=42
ρ = 3.2 g/cm^3  E = 400GPa   ν = 0.35
length scale cm
"""
prop0 = Dict("name"=> "PlaneStressPlasticity","rho"=> 4.5, "E"=> 100e+10, "nu"=> 0.2,
"sigmaY"=>0.97e+10, "K"=>5.0e+10)
prop1 = Dict("name"=> "PlaneStress", "rho"=> 3.2, "E"=>400e10, "nu"=>0.35)
#prop1 = prop0
# testtype = "PlaneStress"
# prop = Dict("name"=> testtype, "rho"=> 8000.0, "E"=> 200e+9, "nu"=> 0.45)

include("NNPlatePull_Domain_Data.jl")


domain = Domain(nodes, elements, ndofs, EBC, g, FBC, fext)
state = zeros(domain.neqs)
∂u = zeros(domain.neqs)
globdat = GlobalData(state,zeros(domain.neqs), zeros(domain.neqs),∂u, domain.neqs, gt, ft)

assembleMassMatrix!(globdat, domain)
updateStates!(domain, globdat)


for i = 1:NT
    @info i, "/" , NT
    solver = NewmarkSolver(Δt, globdat, domain, -1.0, 0.0, 1e-4, 1e-6, 10)
    # close("all")
    # visσ(domain,-1.5e9, 4.5e9)
    # savefig("Debug/$i.png")
    # error()
    if i==75
        close("all")
        visσ(domain)
        # visσ(domain,-1.5e9, 4.5e9)
        savefig("Debug/terminal$(tid)i=75.png")
    end
end

# error()
# todo write data
write_data("$(@__DIR__)/Data/$tid.dat", domain)
# plot
close("all")
scatter(nodes[:, 1], nodes[:,2], color="red")
u,v = domain.state[1:domain.nnodes], domain.state[domain.nnodes+1:end]
scatter(nodes[:, 1] + u, nodes[:,2] + v, color="blue")

close("all")
visσ(domain)
savefig("Debug/terminal$tid.png")

@save "Data/domain$tid.jld2" domain
