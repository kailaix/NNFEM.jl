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
  tf.gradients(φ, ε)
end

# parameter
c1 = Variable(0.5) * 1e-1
c2 = Variable(0.5) * 1e-1
function nn_law(ε)
  compute_stress_rivlin_saunders(ε, c1, c2)
end


d, v, a= ExplicitSolver(globaldata, domain, d0, v0, a0, Δt, NT, nn_law, Fext, ubd, abd; strain_type="finite")


idx = @. (0:n)*m + 1
idx = [idx; (@. idx + domain.nnodes)]
loss = sum((d[:, idx] - d_[:, idx])^2)
sess = Session(); init(sess)
@info run(sess, loss)
# error()

# BFGS!(sess, loss)


# for visualizing the parameters 
sol = []
cb = (vs, iter, loss)->begin 
  push!(sol, vs)
  printstyled("[#iter $iter] a = $vs, loss=$loss\n", color=:green)
end
BFGS!(sess, loss, callback = cb, vars = [c1, c2])

plot([x[1] for x in sol], "+--", color="red", label="\$C1\$")
plot(ones(length(sol))*1e-1, "-", color="k", alpha=0.3, label="Reference Value (C1 and C2)")
plot([x[2] for x in sol], "x--", color ="green", label="\$C2\$")
xlabel("Iterations")
legend()

