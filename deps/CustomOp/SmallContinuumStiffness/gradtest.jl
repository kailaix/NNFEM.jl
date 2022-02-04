using Revise
using NNFEM
using PoreFlow
using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
Random.seed!(233)

function small_continuum_stiffness(k)
    small_continuum_stiffness_ = load_op_and_grad("./build/libSmallContinuumStiffness","small_continuum_stiffness", multiple=true)
    k = convert_to_tensor([k], [Float64]); k = k[1]
    ii, jj, vv = small_continuum_stiffness_(k)
    SparseTensor(ii+1, jj+1, vv, 2*domain.nnodes, 2*domain.nnodes)
end

m = 10
n = 10
h = 0.1
domain = example_domain(m, n, h)
# TODO: specify your input parameters
k = rand(domain.neles * length(domain.elements[1].weights),3,3)
q = copy(k)
for i = 1:4
    q[i,:,:] = k[i,:,:]'
end
init_nnfem(domain)
u = small_continuum_stiffness(k)
sess = Session(); init(sess)
@show run(sess, u)-run(sess, compute_fem_stiffness_matrix(constant(k), m, n, h))
@show 

# uncomment it for testing gradients
# error() 


# TODO: change your test parameter to `m`
#       in the case of `multiple=true`, you also need to specify which component you are testings
# gradient check -- v
function scalar_function(m)
    return sum(values(small_continuum_stiffness(m))^2)
end

# TODO: change `m_` and `v_` to appropriate values
m_ = constant(rand(domain.neles * length(domain.elements[1].weights),3,3))
v_ = rand(domain.neles * length(domain.elements[1].weights),3,3)
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
