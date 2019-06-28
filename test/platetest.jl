using Revise
using Test 
using NNFEM
using PyCall
using PyPlot

elements_, nodes, boundaries = readMesh("$(@__DIR__)/../deps/plate.msh")
# Dirichlet_1 : bottom
# Dirichlet_2 : right
# Dirichlet_3 : top 
# Dirichlet_4 : left
function set_boundary(boundaries, nnodes, ndofs = 2)
    EBC, g = zeros(Int64, nnodes, ndofs), zeros(nnodes, ndofs)
    NBC, f = zeros(Int64, nnodes, ndofs), zeros(nnodes, ndofs)
    for (boundaryname, boundaryedges) in boundaries
        if boundaryname == "\"Dirichlet_1\""
            for (n1, n2) in boundaryedges
                EBC[n1, 2] = -1
                g[n1, 2] = 0.0

                EBC[n2, 2] = -1
                g[n2, 2] = 0.0
            end
            
            
        elseif boundaryname=="\"Dirichlet_2\""
        elseif boundaryname=="\"Dirichlet_3\""
            for (n1, n2) in boundaryedges
                EBC[n1, 2] = -2
                EBC[n2, 2] = -2
            end
            
        elseif boundaryname=="\"Dirichlet_4\""
            for (n1, n2) in boundaryedges
                EBC[n1, 1] = -1
                g[n1, 1] = 0.0
                EBC[n2, 1] = -1
                g[n1, 1] = 0.0
            end
                
        end
    end
    gt = t -> t*0.01*ones(sum(EBC.==-2))
    
    return EBC, g, NBC, f, gt
end




testtype = "PlaneStress"
ndofs = 2

nnodes = size(nodes,1)
EBC, g ,NBC, f, gt = set_boundary(boundaries, nnodes)

# prop = Dict("name"=> testtype, "rho"=> 8000.0, "E"=> 200e9, "nu"=> 0.45,
#             "sigmaY"=>300e6, "K"=>100)

prop = Dict("name"=> testtype, "rho"=> 8000.0e-9, "E"=> 200, "nu"=> 0.45,
            "sigmaY"=>0.3, "K"=>1/9*200)

# prop = Dict("name"=> testtype, "rho"=> 0.8, "E"=> 20000, "nu"=> 0.45,
#             "sigmaY"=>300, "K"=>10)

elements = []
for i = 1:length(elements_)
        elnodes = elements_[i]
        coords = nodes[elnodes,:]
        push!(elements,SmallStrainContinuum(coords,elnodes, prop))
end

domain = Domain(nodes, elements, ndofs, EBC, g, NBC, f)
state = zeros(domain.neqs)
∂u = zeros(domain.neqs)
globdat = GlobalData(state,zeros(domain.neqs),
                    zeros(domain.neqs),∂u, domain.neqs, gt)
assembleMassMatrix!(globdat, domain)
updateStates!(domain, globdat)

# #@show "F - F1", F - F1
# #@show "F", F
# #@show "K", K
# #@show "M", globdat.M


# solver = ExplicitSolver(Δt, globdat, domain )
NT = 20
Δt = 1/NT
for i = 1:NT
    solver = NewmarkSolver(Δt, globdat, domain, 1.0, 0.5, 1e-6, 10)
end
# visdynamic(domain,"dym")
# solver = StaticSolver(globdat, domain )
#solver.run( props , globdat )
visstatic(domain)

function ct(x)
    if x<0.25
        return sqrt(0.25^2-x^2)
    else
        return 0.0
    end
end
a = LinRange{Float64}(0.0,0.5,100)
y = ct.(a)
plot(a, y, "k--")
plot([0.5;0.5;0.0;0.0],[0.0;0.5;0.5;0.25],"k--")
# end