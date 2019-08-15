using Revise
using Test 
using NNFEM
using PyCall
using PyPlot

np = pyimport("numpy")
nx, ny =  10,10
nnodes, neles = (nx + 1)*(ny + 1), nx*ny
x = np.linspace(0.0, 1.0, nx + 1)
y = np.linspace(0.0, 1.0, ny + 1)
X, Y = np.meshgrid(x, y)
nodes = zeros(nnodes,2)
nodes[:,1], nodes[:,2] = X'[:], Y'[:]
ndofs = 2

EBC, g = zeros(Int64, nnodes, ndofs), zeros(nnodes, ndofs)


# EBC[collect(1:nx+1:(nx+1)*(ny+1)), :] .= -1
# EBC[collect(nx+1:nx+1:(nx+1)*(ny+1) + nx), 2] .= -1
# EBC[collect(nx+1:nx+1:(nx+1)*(ny+1) + nx), 1] .= -1
# g[collect(nx+1:nx+1:(nx+1)*(ny+1) + nx), 1] .= 0.04
# gt = t->0.0

EBC[collect(1:nx+1:(nx+1)*(ny+1)), :] .= -1
# EBC[collect(nx+1:nx+1:(nx+1)*(ny+1) + nx), 2] .= -1
# EBC[collect(nx+1:nx+1:(nx+1)*(ny+1) + nx), 1] .= -2
# gt = t -> t*0.01*ones(sum(EBC.==-2))
gt = t->0.0

# EBC[collect(1:nx+1:(nx+1)*(ny+1)), 1] .= -2
# EBC[collect(1:nx+1:(nx+1)*(ny+1)), 2] .= -1
# EBC[collect(nx+1:nx+1:(nx+1)*(ny+1) + nx), :] .= -1
# gt = t -> -t*0.04*ones(sum(EBC.==-2))

NBC, f = zeros(Int64, nnodes, ndofs), zeros(nnodes, ndofs)


testtype = "PlaneStressPlasticity"
prop = Dict("name"=> testtype, "rho"=> 1.0, "E"=> 2000, "nu"=> 0.3,
            "sigmaY"=>100, "K"=>500)


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




NT = 10
Δt = 1/NT
d = (ny+1)*nx
globdat.velo[1:d] .= 0.1
for i = 1:NT
    solver = NewmarkSolver(Δt, globdat, domain, 0.5, 0.5, 1e-6, 10)
end
# visdynamic(domain,"dym")
# solver = StaticSolver(globdat, domain )
#solver.run( props , globdat )
visstatic(domain)

# function ct(x)
#     if x<0.25
#         return sqrt(0.25^2-x^2)
#     else
#         return 0.0
#     end
# end
# a = LinRange{Float64}(0.0,0.5,100)
# y = ct.(a)
# plot(a, y, "k--")
plot([0.0;1.0;1.0;0.0;0.0],[0.0;0.0;1.0;1.0;0.0],"k--")
# end