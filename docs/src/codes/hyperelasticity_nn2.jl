include("hyperelasticity.jl")

ts = ExplicitSolverTime(Δt, NT)
ubd, abd = compute_boundary_info(domain, globaldata, ts)
Fext = compute_external_force(domain, globaldata, ts) 

d0 = zeros(2domain.nnodes)
v0 = zeros(2domain.nnodes)
a0 = zeros(2domain.nnodes)

# general neural network 
function nn_law(ε)
  ae(ε, [20,20,20,3])
end

# free energy 
function nn_law(ε)
  φ = squeeze(ae(ε, [20,20,20,1]))
  tf.gradients(φ, ε)[1] 
end

# # parameter
# c1 = Variable(0.5) * 1e-1
# c2 = Variable(0.5) * 1e-1
# function nn_law(ε)
#   compute_stress_rivlin_saunders(ε, c1, c2)
# end


d, v, a= ExplicitSolver(globaldata, domain, d0, v0, a0, Δt, NT, nn_law, Fext, ubd, abd; strain_type="finite")


idx = @. (0:n)*m + 1
idx = [idx; (@. idx + domain.nnodes)]
loss = sum((d[:, idx] - d_[:, idx])^2)
sess = Session(); init(sess)
@info run(sess, loss)
# error()

for i = 1:100
  BFGS!(sess, loss, 1000)
  ADCME.save(sess, "data2_$i.mat")
end
