using Revise
using Test 
using MAT
using NNFEM
using PyCall
using PyPlot
using JLD2
using ADCME
using LinearAlgebra


include("nnutil.jl")
testtype = "NeuralNetwork1D"
include("NNTrussPull_Domain.jl")
tid = 3

#nntype = "ae_scaled"
#nntype = "piecewise"
ite_id = 3
nnname = "Data/$(nntype)/trained_nn$(idx)_from3_ite$(ite_id).mat"
#nnname = "Data/$(nntype)/NNPreLSfit_nn$(idx)_ite$(start_id).mat"
#nnname = "Data/$(nntype)/learned_nn$(idx)_ite$(ite_id).mat"
@show nnname
s = ae_to_code(nnname, nntype)
eval(Meta.parse(s))






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


# for i = 1:NT
#     @info i, "/" , NT
#     solver = NewmarkSolver(Δt, globdat, domain, -1.0, 0.0, 1e-8, 1e-8, 10) # ok
# end


domain_te = domain 
@info tid
@save "Data/$(nntype)/domain_te$(tid).jld2" domain_te


############################## Plot stress

close("all")
strain = hcat(domain_te.history["strain"]...)
stress = hcat(domain_te.history["stress"]...)
i = 8
plot(strain[i,:], stress[i,:], ".", label="Estimated")




@load "Data/domain$tid.jld2" domain 
strain = hcat(domain.history["strain"]...)
stress = hcat(domain.history["stress"]...)
i = 8
plot(strain[i,:], stress[i,:], "--", label="Reference")




xlabel("Strain")
ylabel("Stress")
legend()
#mpl.save("truss1d_stress$tid.tex")
savefig("nntest$(idx)_$(nntype)_truss1d_stress$tid.png")