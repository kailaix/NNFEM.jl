using Revise
using Test 
using NNFEM
using PyCall
using PyPlot
using JLD2
using ADCME
using LinearAlgebra


np = pyimport("numpy")
nx, ny =  8,10
nnodes, neles = (nx + 1)*(ny + 1), nx*ny
x = np.linspace(0.0, 0.5, nx + 1)
y = np.linspace(0.0, 0.5, ny + 1)
X, Y = np.meshgrid(x, y)
nodes = zeros(nnodes,2)
nodes[:,1], nodes[:,2] = X'[:], Y'[:]
ndofs = 2

EBC, g = zeros(Int64, nnodes, ndofs), zeros(nnodes, ndofs)

EBC[collect(1:nx+1), :] .= -1

function ggt(t)
    return zeros(sum(EBC.==-2)), zeros(sum(EBC.==-2))
end
gt = ggt



#pull in the y direction
FBC, fext = zeros(Int64, nnodes, ndofs), zeros(nnodes, ndofs)


# todo PARAMETER
FORCE_TYPE = "constant"

if FORCE_TYPE == "constant"
    #pull in the y direction
    FBC[collect((nx+1)*ny + 1:(nx+1)*ny + nx+1), 2] .= -1
    fext[collect((nx+1)*ny + 1:(nx+1)*ny + nx+1), 2] = collect(range(2.0, stop=5.0, length=nx+1))*1e8*(0.1*tid+0.5)
    fext[(nx+1)*ny + 1, 2] /= 2.0
    fext[(nx+1)*ny + nx+1, 2] /= 2.0
else
    FBC[collect((nx+1)*ny + 1:(nx+1)*ny + nx+1), 2] .= -2
end

#force load function
function fft(t)
    f = 1.0e8 *(1.0*tid) * sin(pi*t/T) * ones(nx + 1)
    f[1] /= 2.0
    f[end] /= 2.0
    return f
end
ft = fft

elements = []
for j = 1:ny
    for i = 1:nx 
        n = (nx+1)*(j-1) + i
        elnodes = [n, n + 1, n + 1 + (nx + 1), n + (nx + 1)]
        coords = nodes[elnodes,:]
        push!(elements,SmallStrainContinuum(coords,elnodes, prop,2))
    end
end

T = 0.0005
NT = 100
Δt = T/NT
stress_scale = 1.0e10
strain_scale = 1