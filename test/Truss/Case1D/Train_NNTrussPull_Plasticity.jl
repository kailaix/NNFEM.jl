#= 


=#


using Revise
using Test 
using NNFEM
using PyCall
using PyPlot
using JLD2
using ADCME
using LinearAlgebra
reset_default_graph()


include("nnutil.jl")
include("NNTrussPull_Domain.jl")
testtype = "NeuralNetwork1D"
prop = Dict("name"=> testtype, "rho"=> 8000.0, "E"=> 200e3, "nu"=> 0.45,
           "sigmaY"=>0.3e3, "K"=>1/9*200e3, "B"=> 0.0, "A0"=> 0.005, "nn"=>post_nn)

n_data = [1,2,4,5]
nodes, EBC, g, gt, FBC, fext, ft, npoints, node_to_point = BoundaryCondition(ndata[1])
elements = []
for i = 1:nx 
    elnodes = [i, i+1]; coords = nodes[elnodes,:];
    push!(elements, FiniteStrainTruss(coords,elnodes, prop, ngp))
end
domain = Domain(nodes, elements, ndofs, EBC, g, FBC, fext)
state = zeros(domain.neqs)
∂u = zeros(domain.neqs)
globdat = GlobalData(state,zeros(domain.neqs), zeros(domain.neqs),∂u, domain.neqs, gt, ft)
assembleMassMatrix!(globdat, domain)




losses = Array{PyObject}(undef, length(n_data))
for (i, tid) in enumerate(n_data)
    state_history, fext_history = read_data("$(@__DIR__)/Data/$tid.dat")
    losses[i] = DynamicMatLawLoss(domain, globdat, state_history, fext_history, nn,Δt)
end
loss = sum(losses)


sess = Session(); init(sess)
@show run(sess, loss)

if !isdir("Data/$(nntype)")
    mkdir("Data/$(nntype)")
end

start_id = 3
ADCME.load(sess, "Data/$(nntype)/NNPreLSfit_$(start_id).mat")
for i = 1:10
    println("************************** Outer Iteration = $i ************************** ")
    BFGS!(sess, loss, 2000)
    ADCME.save(sess, "Data/$(nntype)/trained_from$(start_id)_ite$(i).mat")
end




# X, Y = prepare_strain_stress_data1D(domain)
# x = (constant(X[:,1]), constant(X[:,2]), constant(X[:,3]))
# y = squeeze(nn(x...))
# close("all")
# out = run(sess, y)
# plot(X[:,1], out,"+", label="NN")
# plot(X[:,1], Y, ".", label="Exact")
# legend()
