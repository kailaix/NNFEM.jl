using Revise
using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
using NNFEM
Random.seed!(233)

# function viscoelasticity_matrices(mu,eta,lambda,dt)
#     viscoelasticity_matrices_ = load_op_and_grad("./build/libViscoelasticityMatrices","viscoelasticity_matrices", multiple=true)
#     mu,eta,lambda,dt = convert_to_tensor(Any[mu,eta,lambda,dt], [Float64,Float64,Float64,Float64])
#     viscoelasticity_matrices_(mu,eta,lambda,dt)
# end

# TODO: specify your input parameters
domain = example_domain(10,10,0.1)
init_nnfem(domain)
N = getNGauss(domain)
mu = rand(N)
eta = rand(N)
lambda = rand(N)
dt = 0.01
S = zeros(N,3,3)
H = zeros(N,3,3)
for i = 1:N 
    p = mu[i] * dt / eta[i]
    S[i,:,:] = inv(
        [
            1+2/3*p -1/3*p 0.0
            -1/3*p 1+2/3*p 0.0
            0.0 0.0 1+p 
        ]
    )
    H[i,:,:] = S[i,:,:] * [
        2*mu[i] + lambda[i] lambda[i] 0.0
        lambda[i] 2mu[i] + lambda[i] 0.0
        0.0 0.0 mu[i]
    ]
end
# s, h = viscoelasticity_matrices(mu,eta,lambda,dt)
s, h = compute_maxwell_viscoelasticity_matrices(mu,lambda,eta,dt)

sess = Session(); init(sess)
@show S0, H0 = run(sess, [s,h])
@show maximum(abs.(S0-S))
@show maximum(abs.(H0-H))

# uncomment it for testing gradients
# error() 


# TODO: change your test parameter to `m`
#       in the case of `multiple=true`, you also need to specify which component you are testings
# gradient check -- v
function scalar_function(m)
    # return sum(viscoelasticity_matrices(mu,eta,lambda,dt)[2]^2)
    return sum(compute_maxwell_viscoelasticity_matrices(mu,eta,m,dt)[2]^2)
end

# TODO: change `m_` and `v_` to appropriate values
m_ = constant(rand(N))
v_ = rand(N)
y_ = scalar_function(m_)
dy_ = gradients(y_, m_)
ms_ = Array{Any}(undef, 5)
ys_ = Array{Any}(undef, 5)
s_ = Array{Any}(undef, 5)
w_ = Array{Any}(undef, 5)
gs_ =  @. 1 / 10^(1:5)

for i = 1:5
    g_ = gs_[i]
    ms_[i] = m_ + g_*v_
    ys_[i] = scalar_function(ms_[i])
    s_[i] = ys_[i] - y_
    w_[i] = s_[i] - g_*sum(v_.*dy_)
end

sess = Session(); init(sess)
sval_ = run(sess, s_)
wval_ = run(sess, w_)
close("all")
loglog(gs_, abs.(sval_), "*-", label="finite difference")
loglog(gs_, abs.(wval_), "+-", label="automatic differentiation")
loglog(gs_, gs_.^2 * 0.5*abs(wval_[1])/gs_[1]^2, "--",label="\$\\mathcal{O}(\\gamma^2)\$")
loglog(gs_, gs_ * 0.5*abs(sval_[1])/gs_[1], "--",label="\$\\mathcal{O}(\\gamma)\$")

plt.gca().invert_xaxis()
legend()
xlabel("\$\\gamma\$")
ylabel("Error")
