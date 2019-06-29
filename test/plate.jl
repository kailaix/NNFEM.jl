# @testset "workflow" begin
using Revise
using Test 
using NNFEM
using PyPlot
using PyCall

testtype = "PlaneStressPlasticity"
np = pyimport("numpy")
nx, ny =  1,4
nnodes, neles = (nx + 1)*(ny + 1), nx*ny
x = np.linspace(0.0, 0.5, nx + 1)
y = np.linspace(0.0, 0.5, ny + 1)
X, Y = np.meshgrid(x, y)
nodes = zeros(nnodes,2)
nodes[:,1], nodes[:,2] = X'[:], Y'[:]
ndofs = 2

EBC, g = zeros(Int64, nnodes, ndofs), zeros(nnodes, ndofs)

EBC[collect(1:nx+1), :] .= -1
EBC[collect((nx+1)*ny + 1:(nx+1)*ny + nx+1), 1] .= -1
#pull in the y direction
EBC[collect((nx+1)*ny + 1:(nx+1)*ny + nx+1), 2] .= -2
one_dim = true
if one_dim
    EBC[collect((nx+1) : nx+1: (nx+1)*(ny+1)), 1] .= -1
    EBC[collect(1 : nx+1: (nx+1)*ny+1), 1] .= -1
end
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



prop = Dict("name"=> testtype, "rho"=> 8000.0, "E"=> 200e+9, "nu"=> 0.45,
"sigmaY"=>0.3e+9, "K"=>1/9*200e+9)

elements = []
for j = 1:ny
    for i = 1:nx 
        n = (nx+1)*(j-1) + i
        elnodes = [n, n + 1, n + 1 + (nx + 1), n + (nx + 1)]
        coords = nodes[elnodes,:]
        push!(elements,SmallStrainContinuum(coords,elnodes, prop))
    end
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
visstatic(domain)

#solver.run( props , globdat )
# visualize(domain)
# end