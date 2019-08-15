using Revise
using Test 
using NNFEM
using PyCall
using PyPlot
using JLD2
using ADCME
using LinearAlgebra

nntype = "linear"
H = Variable(diagm(0=>ones(3)))
# H = Variable(rand(3,3))
H = constant([2.50784e11 1.12853e11 0.0; 1.12853e11 2.50784e11 0.0; 0.0 0.0 6.89655e10])

function nn(ε, ε0, σ0)
    local y
    if nntype=="linear"
        y = ε*H
    elseif nntype=="nn"
        x = [ε ε0 σ0]
        y = ae(x, [20,20,20,20,3], "nn")
    end
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
# assembleMassMatrix!(globdat, domain)
# updateStates!(domain, globdat)


# # test on plate with hole
# neles = domain.neles
# nGauss = length(domain.elements[1].weights)
# nstrains = 3
# NT = 10
# E_all = (rand(NT+1, neles*nGauss, nstrains))
# fext = (rand(NT, domain.neqs))
# loss = DynamicMatLawLoss(domain, E_all, fext, nn)

# sess = Session(); init(sess)
# run(sess, loss)

# MinimizeLoss(sess, loss)
include("$(@__DIR__)/plate.jl")
close("all");
Fext, E_all = preprocessing(domain, globdat, zeros(domain.neqs, NT+1), Δt)
loss = DynamicMatLawLoss(domain, E_all, Fext, nn)
sess = Session(); init(sess)
@show run(sess, loss)
# ADAM(sess, loss)