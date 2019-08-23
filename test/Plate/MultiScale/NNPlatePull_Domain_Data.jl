if Sys.MACHINE=="x86_64-apple-darwin18.6.0"
    matplotlib.use("macosx")
end

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
fiber_size = 2
ndofs = 2
nxc, nyc = 40,20
nx, ny =  nxc*fiber_size, nyc*fiber_size
#Type 1=> SiC, type 0=>Ti, each fiber has size is k by k
fiber_fraction = 0.25
fiber_distribution = "Uniform"
ele_type = generateEleType(nxc, nyc, fiber_size, fiber_fraction, fiber_distribution)
@show "fiber fraction: ",  sum(ele_type)/(nx*ny)
# matshow(ele_type)
# savefig("test.png")
# error()

F0 = 5e5
nodes, EBC, g, gt, FBC, fext, ft = BoundaryCondition(tid, nx, ny, F0)
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

T = 0.00005
NT = 100
Δt = T/NT
stress_scale = 1.0e5
strain_scale = 1
