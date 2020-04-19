include("hyperelasticity.jl")
# 
ts = ExplicitSolverTime(Δt, NT)
ubd, abd = compute_boundary_info(domain, globaldata, ts)
Fext = compute_external_force(domain, globaldata, ts) 

d0 = zeros(2domain.nnodes)
v0 = zeros(2domain.nnodes)
a0 = zeros(2domain.nnodes)


mode = "linear"
if length(ARGS)>=1
  global mode = ARGS[1]
end

if mode=="linear"
  global Dc, nn_law
  Dc = Variable(rand(3,3))
  Dc = spd(Dc)
  function nn_law(strain)
    strain_tensor = strain_voigt_to_tensor(strain)
    stress = strain*Dc
    stress
  end

elseif mode=="consistent_tangent"
  global Dc, nn_law
  Dc = Variable(rand(3,3))
  Dc = spd(Dc)
  function nn_law(strain)
    coef = ae(strain, [20,20,20,6])
    coef = [coef constant(ones(size(coef,1), 1))]
    H = consistent_tangent_matrix(coef, Dc)
    stress = batch_matmul(H, strain)
    stress
  end

elseif mode=="nn"
  global Dc, nn_law
  # general neural network 
  function nn_law(ε)
    ae(ε, [20,20,20,3])
  end
elseif mode=="free_energy"
  global Dc, nn_law
  # free energy 
  function nn_law(ε)
    φ = squeeze(ae(ε, [20,20,20,1]))
    tf.gradients(φ, ε)[1]
  end
else 
  error("$mode not valid")
end

d, v, a= ExplicitSolver(globaldata, domain, d0, v0, a0, Δt, NT, nn_law, Fext, ubd, abd; strain_type="finite")

# test  = open("test.txt", "w")
sess = Session(); init(sess)
# run(sess, d)
idx = @. (0:n)*m + 1
idx = [idx; (@. idx + domain.nnodes)]
loss = sum((d[:, idx] - d_[:, idx])^2)
sess = Session(); init(sess)
@info run(sess, loss)
# error()

!isdir(mode) && mkdir(mode)

for i = 1:100
  loss_ = BFGS!(sess, loss, 1000)
  d0 = run(sess, d)

  # visualize
  close("all")
  p = visualize_displacement(d0, domain)
  saveanim(p, "$mode/$i.gif")
  close("all")
  plot(d_[:,1], "-", color="C1")
  plot(d0[:,1], "--", color="C1")
  plot(d_[:,1+domain.nnodes], color="C2")
  plot(d0[:,1+domain.nnodes], "--", color="C2")
  savefig("$mode/$i.png")

  open("$mode/loss.txt", "a") do io 
    writedlm(io, loss_)
  end  
  ADCME.save(sess, "$mode/$i.mat")
  if length(loss_)<1000
    break
  end
end
