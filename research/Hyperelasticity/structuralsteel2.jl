include("structuralsteel1.jl")


ts = ExplicitSolverTime(Δt, NT)
ubd, abd = compute_boundary_info(domain, globaldata, ts)
Fext = compute_external_force(domain, globaldata, ts) 



d0 = zeros(2domain.nnodes)
v0 = zeros(2domain.nnodes)
a0 = zeros(2domain.nnodes)
σ0 = zeros(getNGauss(domain),3)
ε0 = zeros(getNGauss(domain),3)


function nn_law(ε, εc, σc)
    coef = ae(ε, [20,20,20,6]) 
    H = spd_Cholesky(coef) 
    stress = batch_matmul(H, ε-εc) * 1e4 + σc
    stress
end

d, v, a= ExplicitSolver(globaldata, domain, d0, v0, a0, σ0, ε0, Δt, NT, nn_law, Fext, ubd, abd; strain_type="finite")

d_ = matread("data/3.mat")["d"]

using Random; Random.seed!(2333)
idx = sample_interior(domain.nnodes, 100)
loss = mean((d[:,idx] - d_[1:10:end, idx])^2)*1e10

# # test  = open("test.txt", "w")
sess = Session(); init(sess)
# run(sess, loss)
BFGS!(sess, loss, 1000)
d_ = run(sess, d)
visualize_total_deformation_on_scoped_body(d_, domain;scale_factor=50.)

