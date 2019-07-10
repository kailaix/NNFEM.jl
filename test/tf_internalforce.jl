using Revise
using Test 
using NNFEM
using PyCall
using PyPlot
using JLD2
using ADCME

function nn(ε, ε0, σ0)
    @show [constant([ε ε0]) σ0]
    x = [constant([ε ε0]) σ0]
    y = ae(x, [20,20,20,20,3], "nn")
    @show y
    y
end

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
    function ggt(t)
        v = 0.01
        if t<1.0
            t*v*ones(sum(EBC.==-2))
        elseif t<3.0
            (0.02 - t*v)*ones(sum(EBC.==-2))
        end
    end
    gt = ggt
    
    return EBC, g, NBC, f, gt
end




testtype = "NeuralNetwork2D"
ndofs = 2

nnodes = size(nodes,1)
EBC, g ,NBC, f, gt = set_boundary(boundaries, nnodes)

prop = Dict("name"=> testtype, "rho"=> 8000.0, "E"=> 200e9, "nu"=> 0.45,
            "sigmaY"=>300e6, "K"=>1/9*200e9, "nn"=>nn)

# prop = Dict("name"=> testtype, "rho"=> 8000.0e-9, "E"=> 200, "nu"=> 0.45,
#             "sigmaY"=>0.3, "K"=>1/9*200)

# prop = Dict("name"=> testtype, "rho"=> 1.0, "E"=> 2000, "nu"=> 0.3,
#             "sigmaY"=>100, "K"=>500)

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

fint = tfAssembleInternalForce(globdat, domain,nn)