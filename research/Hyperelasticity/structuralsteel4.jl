#=
Differentiable moley-rivlin 
=#
include("structuralsteel1.jl")



ts = ExplicitSolverTime(Δt, NT)
ubd, abd = compute_boundary_info(domain, globaldata, ts)
Fext = compute_external_force(domain, globaldata, ts) 



d0 = zeros(2domain.nnodes)
v0 = zeros(2domain.nnodes)
a0 = zeros(2domain.nnodes)
ε0 = zeros(getNGauss(domain), 3)
σ0 = zeros(getNGauss(domain), 3)

c1 = prop["C1"]
c2 = prop["C2"]

# "C1"=>3.3017e5, "C2"=>30515.0
c1 = 1e4
c2 = 1e4
function nn_law(ε, εc, σc)
    ε1 = 1e4 * ε
    Δσ = fc(ε1, [20,20,20,3])*100
    compute_stress_rivlin_saunders(ε, c1, c2) + Δσ
end

d, v, a, σ, ε = ExplicitSolver(globaldata, domain, d0, v0, a0, ε0, σ0, Δt, NT, nn_law, Fext, ubd, abd; strain_type="finite")

d_ = matread("data/3.mat")["d"]

using Random; Random.seed!(2333)
idx = sample_interior(domain.nnodes, 100)
loss = mean((d[:,idx] - d_[:, idx])^2)*1e10

# # test  = open("test.txt", "w")
sess = Session(); init(sess)
@info run(sess, loss)
# BFGS!(sess, loss, 1000)
σ_, d_ = run(sess, [σ, d])
for i = 1:size(σ_,1)
    push!(domain.history["stress"], σ_[i,:,:])
end
p = visualize_von_mises_stress_on_scoped_body(d_, domain;scale_factor=50.)

