using Revise
using Test 
using MAT
using NNFEM
using PyCall
using PyPlot
using JLD2
using ADCME
using LinearAlgebra


#nntype = "ae_scaled"
nntype = "piecewise"
ite_id = 10
nnname = "Data/trained_$(nntype)_ite$(ite_id).mat"
s = ae_to_code(nnname, nntype)

eval(Meta.parse(s))

include("nnutil.jl")


# testtype = "PlaneStressPlasticity"
testtype = "NeuralNetwork1D"
include("NNTrussPull_Domain.jl")


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


# for i = 1:NT
#     @info i, "/" , NT
#     solver = NewmarkSolver(Δt, globdat, domain, -1.0, 0.0, 1e-5, 1e-5, 10) # ok
# end


domain_te = domain 
@info tid
@save "Data/domain_$(nntype)_te$(tid).jld2" domain_te