using Revise
using ADCME
using NNFEM
using JLD2
using PyPlot
using MAT 
using DelimitedFiles
reset_default_graph()

include("nnutil.jl")

loss = constant(0.0)
for tid = [1,2,4,5]
    strain, stress = read_strain_stress("Data/$(tid).dat")
    X, Y = prepare_strain_stress_data1D(strain, stress )
    y = squeeze(nn(constant(X[:,1:1]), constant(X[:,2:2]), constant(X[:,3:3])))
    global loss += sum((y - Y)^2)
end
sess = Session(); init(sess)
@show run(sess,loss)

loss_all = []
for i = 1:10
    println("************************** Outer Iteration = $i ************************** ")
    loss_ = BFGS!(sess, loss, 3000)
    push!(loss_all, loss_)
end 

testtype = "NeuralNetwork1D"
include("NNTrussPull_Domain.jl")
tid = 3

nodes, EBC, g, gt, FBC, fext, ft, npoints, node_to_point = BoundaryCondition(tid)
prop = Dict("name"=> testtype, "rho"=> 8000.0, "E"=> 200e3, "nu"=> 0.45,
           "sigmaY"=>0.3e3, "K"=>1/9*200e3, "B"=> 0.0, "A0"=> 0.005, "nn"=>post_nn)

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
# need to update state in domain from globdat
updateStates!(domain, globdat)


adaptive_solver_args = Dict("Newmark_rho"=> 0.0, 
                          "Newton_maxiter"=>10, 
                          "Newton_Abs_Err"=>1e-4, 
                          "Newton_Rel_Err"=>1e-6, 
                          "damped_Newton_eta" => 1.0)
globdat, domain, ts = AdaptiveSolver("NewmarkSolver", globdat, domain, T, NT, adaptive_solver_args)

state_ = hcat(domain.history["state"]...)


@load "Data/domain3.jld2" domain 
state = hcat(domain.history["state"]...)
loss = norm(state_ - state)
open("Data/$FILEID2.txt", "a") do io
    writedlm(io, [loss])
end