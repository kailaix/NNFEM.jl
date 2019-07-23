using Revise
using Test 
using NNFEM
using PyCall
using PyPlot
using JLD2
using ADCME
using LinearAlgebra

np = pyimport("numpy")

"""
1  -  2  -  3  -  4


local node id: left to right
"""

ndofs = 2
nx = 4
nnodes, neles = (nx + 1), nx

# PARAMETER: distance between nodes 0-1 and 0-2
l = 1.0 
nodes = zeros(Float64, nnodes, ndofs)
for i = 1:nx+1
    nodes[i, 1], nodes[i, 2] = (i - 1)*l, 0.0
end


EBC, g = zeros(Int64, nnodes, ndofs), zeros(nnodes, ndofs)
#fix node 1 
EBC[1, :] .= -1



function ggt(t)
    return zeros(sum(EBC.==-2)), zeros(sum(EBC.==-2))
end
gt = ggt

#pull in the x direction
FBC, fext = zeros(Int64, nnodes, ndofs), zeros(nnodes, ndofs)


# todo PARAMETER
FORCE_TYPE = "non-constant"



if FORCE_TYPE == "constant"
    FBC[nx+1, 1]  = -1
    fext[nx+1, 1] = 1.0e3
else
    FBC[nx+1, 1] = -2
end

#force load function
function fft(t)
    return 1.e3 * sin(2*pi*t)
end
ft = fft


ngp=2
