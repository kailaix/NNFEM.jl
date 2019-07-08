# @testset "workflow" begin
using Revise
using Test 
using NNFEM
using PyPlot
using PyCall


testtype = "Plasticity1D" #"Elasticity1D"
np = pyimport("numpy")

"""
nx = 3 blocks, node 0 is at the origin
2     4     6     8
|  _  |  _  |  _  |
|  X  |  X  |  X  |
|  _  |  _  |  _  |
1     3     5     7

element number: left, up, bottom, diag, anti-diag, right
2i    2i+2
|  -  |
|  X  |
|  -  |
2i-1  2i+1

local node id: left to right, bottom to top
"""

ndofs = 2
nx = 1
nnodes, neles = 2*(nx + 1), 5*nx + 1

# PARAMETER: distance between nodes 0-1 and 0-2
l = 1.0 
nodes = zeros(Float64, nnodes, ndofs)
for i = 1:nx+1
    nodes[2 * i - 1, 1], nodes[2 * i - 1, 2] = (i - 1)*l, 0.0
    nodes[2 * i, 1], nodes[2 * i, 2] = (i - 1)*l, l
end


EBC, g = zeros(Int64, nnodes, ndofs), zeros(nnodes, ndofs)
#fix node 1 and node 2
EBC[[1,2], :] .= -1
#pull in the y direction
EBC[2*(nx+1), 2] = -2

function ggt(t)
    v = 0.01
    if t<1.0
        t*v*ones(sum(EBC.==-2))
    elseif t<3.0
        (0.02 - t*v)*ones(sum(EBC.==-2))
    end
end
gt = ggt


NBC, f = zeros(Int64, nnodes, ndofs), zeros(nnodes, ndofs)


prop = Dict("name"=> testtype, "rho"=> 0.1, "E"=> 200, "B"=> 10.0,
            "sigmaY"=>0.300, "K"=>1/9*200, "A0"=> 1.0)

elements = []
ngp = 2
for i = 1:nx 
    if i == 1
        elnodes = [2*i-1, 2*i]; coords = nodes[elnodes,:];
        push!(elements, FiniteStrainTruss(coords,elnodes, prop, ngp))
    end
    elnodes = [2*i+1, 2*i+2]; coords = nodes[elnodes,:];
    push!(elements, FiniteStrainTruss(coords,elnodes, prop, ngp))

    elnodes = [2*i-1, 2*i+1]; coords = nodes[elnodes,:];
    push!(elements, FiniteStrainTruss(coords,elnodes, prop, ngp))


    elnodes = [2*i, 2*i+2]; coords = nodes[elnodes,:];
    push!(elements, FiniteStrainTruss(coords,elnodes, prop, ngp))

    elnodes = [2*i-1, 2*i+2]; coords = nodes[elnodes,:];
    push!(elements, FiniteStrainTruss(coords,elnodes, prop, ngp))

    elnodes = [2*i, 2*i+1]; coords = nodes[elnodes,:];
    push!(elements, FiniteStrainTruss(coords,elnodes, prop, ngp))
end


domain = Domain(nodes, elements, ndofs, EBC, g, NBC, f)
state = zeros(domain.neqs)
∂u = zeros(domain.neqs)
globdat = GlobalData(state,zeros(domain.neqs),
                    zeros(domain.neqs),∂u, domain.neqs, gt)

assembleMassMatrix!(globdat, domain)
updateStates!(domain, globdat)



T = 2.0
NT = 20
Δt = T/NT
for i = 1:NT
    @info i, "/" , NT
    solver = NewmarkSolver(Δt, globdat, domain, -1.0, 0.0, 1e-6, 10)
    
end
# solver = StaticSolver(globdat, domain )
#visstatic(domain)
