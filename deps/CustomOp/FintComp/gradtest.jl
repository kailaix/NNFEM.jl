using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
Random.seed!(233)

fint_comp = load_op_and_grad("./build/libFintComp", "fint_comp")
################## End Load Operator ##################

neqs_ = 20
θ = constant(rand(3,3))
A_ = rand(10,3)
fints = A_*θ
el = constant(rand(0:neqs_, 10, 3), dtype=Int32)
neqs = constant(neqs_, dtype=Int32)
# TODO: specify your input parameters
u = fint_comp(fints,el,neqs)
sess = tf.Session()
init(sess)
run(sess, u)
error()

# TODO: change your test parameter to `m`
# gradient check -- v
function scalar_function(θ)
    m = A_*θ
    return sum(tanh(fint_comp(m,el,neqs)))
end

# TODO: change `m_` and `v_` to appropriate values
m_ = constant(rand(3,3))
v_ = rand(3,3)
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

sess = Session()
init(sess)
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
