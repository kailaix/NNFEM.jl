#=
Simulate data
=#
include("structuralsteel1.jl")
assembleMassMatrix!(globaldata, domain)
SolverInitial!(Δt, globaldata, domain)

updateStates!(domain, globaldata)
ω = EigenMode(Δt, globaldata, domain)
@show "stable time step is ", 0.8 * 2/ω, " current time step is ", Δt
for i = 1:NT
    @info i 
    # global globaldata, domain = GeneralizedAlphaSolverStep(globaldata, domain, Δt)
    global globaldata, domain = ExplicitSolverStep(globaldata, domain, Δt)
end
d_ = hcat(domain.history["state"]...)'|>Array


# visualize_total_deformation_on_scoped_body(d_, domain;scale_factor=50.)
visualize_von_mises_stress_on_scoped_body(d_, domain;scale_factor=50.)

matwrite("data/3.mat", Dict(
    "d"=>d_
))