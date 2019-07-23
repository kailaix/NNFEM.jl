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
    v = 0.01
    if t<1.0
        state = t*v*ones(sum(EBC.==-2))
    elseif t<3.0
        state = (0.02 - t*v)*ones(sum(EBC.==-2))
    end
    return state, zeros(sum(EBC.==-2))
end
gt = Nothing

NBC, f = zeros(Int64, nnodes, ndofs), zeros(nnodes, ndofs)

#pull in the x direction
NBC, fext = zeros(Int64, nnodes, ndofs), zeros(nnodes, ndofs)
NBC[nx+1, 1] = -1

#modify this line for new data
fext[nx+1, 1] = 1.0e3

ngp=2
