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
function generateEleType(nxc, nyc, fiber_size, fiber_fraction, fiber_distribution)
    if fiber_distribution == "Uniform"
        nx,ny = nxc*fiber_size , nyc*fiber_size
        # a,fiber_size,a,fiber_size,a void ratio is about fiber_size^2/(a+fiber_size)^2
        ele_type_x =  zeros(Int, nx,1)
        ele_type_y =  zeros(Int, 1,ny)

        a = round(Int, fiber_size*(1.0/sqrt(fiber_fraction)-1))
        if a <2 
            print("Mesh for the multiscale simulaion is too coarse")
            error()
        end
        rx = nx%(a + fiber_size)
        counter = a
        for i = round(Int, (rx+a)/2.0)+1:nx
            if counter%(a+fiber_size) > a-1
                ele_type_x[i] = 1
            end
            counter += 1
        end 

        ry = ny%(a + fiber_size)
        counter = a
        for i = round(Int, (ry+a)/2.0)+1:ny
            if counter%(a+fiber_size) > a-1
                ele_type_y[i] = 1
            end
            counter += 1
        end 
        ele_type = ele_type_x * ele_type_y

    else
        Random.seed!(123)
        d = Binomial(1, fiber_fraction)
        ele_type_c =  rand(d, nxc, nyc)
        ele_type =  kron(ele_type_c, ones(Int, fiber_size, fiber_size))
    end

    return ele_type
end

fiber_size = 2
nxc, nyc = 60,10
nx, ny =  nxc*fiber_size, nyc*fiber_size
#Type 1=> SiC, type 0=>Ti, each fiber has size is k by k
fiber_fraction = 0.25
fiber_distribution = "Uniform"
ele_type = generateEleType(nxc, nyc, fiber_size, fiber_fraction, fiber_distribution)

#matshow(ele_type)
#error()

nnodes, neles = (nx + 1)*(ny + 1), nx*ny
x = np.linspace(0.0, 3.0, nx + 1)
y = np.linspace(0.0, 0.5, ny + 1)


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


# todo PARAMETER

FBC[round(Int,nx*10.0/12.0), 2] = -2


#force load function
function fft(t)
    f = 1.0e4 
end
ft = fft

elements = []
for j = 1:ny
    for i = 1:nx 
        n = (nx+1)*(j-1) + i
        elnodes = [n, n + 1, n + 1 + (nx + 1), n + (nx + 1)]
        coords = nodes[elnodes,:]
        prop = ele_type[i,j] == 0 ? prop0 : prop1
        push!(elements,SmallStrainContinuum(coords,elnodes, prop, 2))
    end
end

T = 0.05
NT = 100
Δt = T/NT
stress_scale = 1.0e5
strain_scale = 1
