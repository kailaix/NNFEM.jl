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
1  -  2  -  3  -  4  -  5


local node id: left to right
"""


ndofs = 2
nx = 4
nnodes, neles = (nx + 1), nx
ngp=2

NT = 1000
Δt = 0.05
T = NT*Δt

function BoundaryCondition(tid::Int64)
# PARAMETER: distance between nodes 0-1 and 0-2
l = 1.0 
nodes = zeros(Float64, nnodes, ndofs)
for i = 1:nx+1
    nodes[i, 1], nodes[i, 2] = (i - 1)*l, 0.0
end

EBC, g = zeros(Int64, nnodes, ndofs), zeros(nnodes, ndofs)
#fix node 1 
EBC[1, :] .= -1
FBC, fext = zeros(Int64, nnodes, ndofs), zeros(nnodes, ndofs)


#tid = 2 # pull from right with force control
#tid = 4 # pull from right with displacement control


gt = nothing 
ft = nothing

if tid == 2
    FBC[nx+1, 1]  = -1
    fext[nx+1, 1] = 1.0e3
elseif tid == 4
    EBC[nx+1, 1] = -2
    EBC[nx+1, 2] = -1
   
    function ggt(t)
         vel = 0.2*1e-3#m/ms
   	 return ones(sum(EBC.==-2))*vel*t, ones(sum(EBC.==-2))*vel, zeros(sum(EBC.==-2))
    end
    gt = ggt
end


npoints = nnodes
node_to_point = collect(1:npoints)

return nodes, EBC, g, gt, FBC, fext, ft, npoints, node_to_point
end



