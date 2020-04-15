using Revise
using Test 
using NNFEM
using PyCall
using PyPlot
using JLD2
using ADCME
using LinearAlgebra
tid = length(ARGS)==1 ? parse(Int64, ARGS[1]) : 3

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
nx = 3
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



function ggt(t)
    return zeros(sum(EBC.==-2)), zeros(sum(EBC.==-2))
end
gt = ggt

#pull in the x direction
FBC, fext = zeros(Int64, nnodes, ndofs), zeros(nnodes, ndofs)

# todo PARAMETER
FORCE_TYPE = "non-constant"

if FORCE_TYPE == "constant"
    #pull in the y direction
    FBC[2*(nx+1), 1] = -1
    fext[2*(nx+1), 1] = 1.0e2
    FBC[2*nx+1, 1] = -1
    fext[2*nx+1, 1] = 1.0e2
else
    FBC[2*(nx+1), 2] = -2
end

#force load function
function fft(t)
    # return 20.0 * sin(2*pi*t)
    return (tid*0.2+0.6)
end
ft = fft

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

# T = 0.5
# NT = 100
T = 0.2
NT = 200

Δt = T/NT
stress_scale = 100.0
