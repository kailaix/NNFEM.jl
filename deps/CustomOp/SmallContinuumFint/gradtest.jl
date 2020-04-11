using Revise
using NNFEM
using PoreFlow
using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
Random.seed!(233)

function small_continuum_fint(stress)
    small_continuum_fint_ = load_op_and_grad("./build/libSmallContinuumFint","small_continuum_fint")
    stress = convert_to_tensor([stress], [Float64]); stress = stress[1]
    small_continuum_fint_(stress)
end

m = 10
n = 10
h = 0.1
domain = example_domain(m, n, h)
init_nnfem(domain)
ngauss = domain.neles * length(domain.elements[1].weights)

stress = rand(ngauss, 3)
# TODO: specify your input parameters
u = small_continuum_fint(stress)
sess = Session(); init(sess)
s = compute_strain_energy_term(stress, m, n, h)
@show maximum(abs.(run(sess, u) - s))

# uncomment it for testing gradients
# error() 


# TODO: change your test parameter to `m`
#       in the case of `multiple=true`, you also need to specify which component you are testings
# gradient check -- v
function scalar_function(m)
    return sum(small_continuum_fint(m)^2)
end

# TODO: change `m_` and `v_` to appropriate values
m_ = constant(rand(ngauss, 3))
v_ = rand(ngauss, 3)
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
