using Revise
using Test 
using NNFEM
using PyCall
using PyPlot
using JLD2
using ADCME
using LinearAlgebra


tid = 203
force_scale=50.0
stress_scale = 1.0

testtype = "NeuralNetwork2D"
nntype = "linear"
include("nnutil.jl")
"""
Property:
The matrix is made of titanium, which are assumed to be elasto-plastic titanium material,  
ρ = 4.5 g/cm^3;  E = 100GPa	 K=10e+9  ν =0.2   σY=970 MPa

Fibers are made of SiC, which are assumed to be isotropic and elastic, with
https://www.azom.com/properties.aspx?ArticleID=42
ρ = 3.2 g/cm^3  E = 400GPa   ν = 0.35
length scale cm
"""
# density 4.5*(1 - 0.25) + 3.2*0.25
fiber_fraction = 0.25
#todo
#fiber_fraction = 1.0
H0 = [1.2671103923007539e6	215999.02498055345	-29.71221633244046
215999.02498055345	690837.2409984174	-2339.4193740513797
-29.71221633244046	-2339.4193740513797	493006.1798000128]
prop = Dict("name"=> testtype, "rho"=> 4.5*(1 - fiber_fraction) + 3.2*fiber_fraction, "nn"=>post_nn)

T = 0.05
NT = 100
Δt = T/NT

nx_f, ny_f = 40*10, 20*10
# nx_f, ny_f = 12, 4

# homogenized computaional domain
# number of elements in each directions
nx, ny = 40, 20
# number of subelements in one element in each directions
sx_f, sy_f = div(nx_f,nx), div(ny_f,ny)

nodes, EBC, g, gt, FBC, fext, ft = BoundaryCondition(tid, nx, ny)

if tid in [100, 300]
    fscale = sx_f
elseif tid in [200, 201, 202, 203]
    fscale = sy_f
end

ft_scale = t->ft(t)*fscale
ndofs=2
elements = []
for j = 1:ny
    for i = 1:nx 
        n = (nx+1)*(j-1) + i
        elnodes = [n, n + 1, n + 1 + (nx + 1), n + (nx + 1)]
        coords = nodes[elnodes,:]
        # 0=> matrix, 1=> fiber
        push!(elements,SmallStrainContinuum(coords,elnodes, prop, 3))
    end
end


domain = Domain(nodes, elements, ndofs, EBC, g, FBC, fext)
state = zeros(domain.neqs)
∂u = zeros(domain.neqs)
globdat = GlobalData(state,zeros(domain.neqs)
, zeros(domain.neqs),∂u, domain.neqs, gt, ft_scale)

assembleMassMatrix!(globdat, domain)
updateStates!(domain, globdat)


for i = 1:NT
    @info i, "/" , NT
    solver = NewmarkSolver(Δt, globdat, domain, -1.0, 0.0, 1e-4, 1e-6, 100)
    # close("all")
    # visσ(domain,-1.5e9, 4.5e9)
    # savefig("Debug/$i.png")
    # error()
    if i==75
        close("all")
        visσ(domain)
        # visσ(domain,-1.5e9, 4.5e9)
        savefig("Debug/test$(tid)i=75.png")
    end
end

# plot
close("all")
scatter(nodes[:, 1], nodes[:,2], color="red")
u,v = domain.state[1:domain.nnodes], domain.state[domain.nnodes+1:end]
scatter(nodes[:, 1] + u, nodes[:,2] + v, color="blue")

close("all")
visσ(domain)
savefig("Debug/test$tid.png")

