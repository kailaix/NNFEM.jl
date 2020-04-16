include("linear_elasticity.jl")


E = Variable(1.0)
ν = Variable(0.0)
H11 = E*(1. -ν)/((1+ν)*(1. -2. *ν));
H12 = H11*ν/(1-ν);
H21 = H12;
H22 = H11;
H33 = H11*0.5*(1. -2. *ν)/(1. -ν);
H = tensor(
  [H11 H12 0.0
  H21 H22 0.0
  0.0 0.0 H33]
)
d, v, a= ExplicitSolver(globaldata, domain, d0, v0, a0, Δt, NT, H, Fext, ubd, abd)
idx = [1;domain.nnodes+1]
loss = sum((d[:, idx] - d_[:, idx])^2)
sess = Session(); init(sess)
BFGS!(sess, loss, var_to_bounds=Dict(E=>(0.0,100.0), ν=>(-0.5,0.49)))
run(sess, [E, ν])
