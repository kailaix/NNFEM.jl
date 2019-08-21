using Revise
using Test 
using NNFEM
using PyCall
using PyPlot
using JLD2
using ADCME
using LinearAlgebra
using Distributions, Random

np = pyimport("numpy")
"""
Feyel, Frédéric, and Jean-Louis Chaboche. 
"FE2 multiscale approach for modelling the elastoviscoplastic behaviour of long fibre SiC/Ti composite materials." 
Computer methods in applied mechanics and engineering 183.3-4 (2000): 309-330.

Geometry:
The plate size is 30 mm by 5mm 
These fibres are made of SiC with diameter about 0.65mm,
The volume fraction is about 33%. 

Domain: 60k by 10k
fibers: k by k
fiber number 200

Property:
Fibers are made of SiC, which are assumed to be isotropic and elastic, with
https://www.azom.com/properties.aspx?ArticleID=42
ρ = 3200 kg/m^3  E = 400GPa   ν = 0.35
The matrix is made of titanium, which are assumed to be elasto-plastic titanium material,  
ρ = 4500 kg/m^3;  E = 100GPa	 K=10e+9  ν =0.2   σY=970 MPa

length scale cm
"""


nx, ny =  30, 5
nnodes, neles = (nx + 1)*(ny + 1), nx*ny
Lx, Ly = 3.0, 0.5
x = np.linspace(0.0, Lx, nx + 1)
y = np.linspace(0.0, Ly, ny + 1)


X, Y = np.meshgrid(x, y)
nodes = zeros(nnodes,2)
nodes[:,1], nodes[:,2] = X'[:], Y'[:]
ndofs = 2

EBC, g = zeros(Int64, nnodes, ndofs), zeros(nnodes, ndofs)

EBC[collect(1:nx+1:(nx+1)*(ny+1)), :] .= -1 # fix left
EBC[collect(nx+1:nx+1:(nx+1)*(ny+1)), 1] .= -1 # symmetric right


function ggt(t)
    return zeros(sum(EBC.==-2)), zeros(sum(EBC.==-2))
end
gt = ggt



#pull in the y direction
FBC, fext = zeros(Int64, nnodes, ndofs), zeros(nnodes, ndofs)


# force parameter
function gauss(L, n, x0; σ=0.2)
    x = collect(LinRange(0, L, n+1))
    g = 1.0/(sqrt(2*pi*σ^2)) * exp.(-0.5*(x .- x0).^2/σ^2)
end


FBC[collect(2:nx+1), 2] .= -1
F = 5e6*(0.2tid)   #elastic 3e6 ; plasticity starts from 4e6 
fext[collect(1:nx+1), 2] = F * gauss(Lx, nx, Lx*5.0/6.0)




#force load function
function fft(t)
    f = 1.0e6 
end
ft = fft

elements = []
for j = 1:ny
    for i = 1:nx 
        n = (nx+1)*(j-1) + i
        elnodes = [n, n + 1, n + 1 + (nx + 1), n + (nx + 1)]
        coords = nodes[elnodes,:]
        prop = ele_type[i,j] == 0 ? prop0 : prop1
        push!(elements,SmallStrainContinuum(coords,elnodes, prop, 3))
    end
end

T = 0.001
NT = 100
Δt = T/NT
stress_scale = 1.0e5
strain_scale = 1
