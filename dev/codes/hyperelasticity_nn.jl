include("hyperelasticity.jl")



ts = ExplicitSolverTime(Δt, NT)
ubd, abd = compute_boundary_info(domain, globaldata, ts)
Fext = compute_external_force(domain, globaldata, ts) 

d0 = zeros(2domain.nnodes)
v0 = zeros(2domain.nnodes)
a0 = zeros(2domain.nnodes)

function nn_law(ε)
  ae(ε, [20,20,20,3])
end
d, v, a= ExplicitSolver(globaldata, domain, d0, v0, a0, Δt, NT, nn_law, Fext, ubd, abd)


idx = @. (0:n)*m + 1
idx = [idx; (@. idx + domain.nnodes)]
loss = sum((d[:, idx] - d_[:, idx])^2)
sess = Session(); init(sess)
BFGS!(sess, loss)
