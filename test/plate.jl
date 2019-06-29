# @testset "workflow" begin
using Revise
using Test 
using NNFEM
using PyPlot
using PyCall

testtype = "PlaneStressPlasticity"
np = pyimport("numpy")
nx, ny =  2,1
nnodes, neles = (nx + 1)*(ny + 1), nx*ny
x = np.linspace(0.0, 1.0, nx + 1)
y = np.linspace(0.0, 1.0, ny + 1)
X, Y = np.meshgrid(x, y)
nodes = zeros(nnodes,2)
nodes[:,1], nodes[:,2] = X'[:], Y'[:]
ndofs = 2

EBC, g = zeros(Int64, nnodes, ndofs), zeros(nnodes, ndofs)

EBC[collect(1:nx+1:(nx+1)*(ny+1)), :] .= -1
EBC[collect(nx+1:nx+1:(nx+1)*(ny+1) + nx), 2] .= -1
EBC[collect(nx+1:nx+1:(nx+1)*(ny+1) + nx), 1] .= -2

function ggt(t)
    v = 0.01
    if t<1.0
        t*v*ones(sum(EBC.==-2))
    elseif t<3.0
        (0.02 - t*v)*ones(sum(EBC.==-2))
    end
end
gt = ggt

# EBC[collect(1:nx+1:(nx+1)*(ny+1)), 1] .= -2
# EBC[collect(1:nx+1:(nx+1)*(ny+1)), 2] .= -1
# EBC[collect(nx+1:nx+1:(nx+1)*(ny+1) + nx), :] .= -1
# gt = t -> -t*0.04*ones(sum(EBC.==-2))

NBC, f = zeros(Int64, nnodes, ndofs), zeros(nnodes, ndofs)



prop = Dict("name"=> testtype, "rho"=> 8000.0e-9, "E"=> 200, "nu"=> 0.45,
"sigmaY"=>0.3, "K"=>1/9*200)

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



T = 2.0
NT = 500
Δt = T/NT
for i = 1:NT
    @show i
    solver = NewmarkSolver(Δt, globdat, domain, -1.0, 0.0, 1e-3, 100)
end
# solver = StaticSolver(globdat, domain )
visstatic(domain)

#solver.run( props , globdat )
# visualize(domain)
# end