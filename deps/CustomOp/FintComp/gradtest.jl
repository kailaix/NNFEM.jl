using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
Random.seed!(233)

py"""
import tensorflow as tf
libFintComp = tf.load_op_library('./build/libFintComp.dylib')
@tf.custom_gradient
def fint_comp(fints,el):
    out = libFintComp.fint_comp(fints,el)
    def grad(dy):
        return libFintComp.fint_comp_grad(dy, out, fints,el)
    return out, grad
"""

fint_comp = py"fint_comp"
################## End Load Operator ##################

fints = constant(rand(10,3))
el = constant(rand(0:30, 10, 3))
# TODO: specify your input parameters
u = fint_comp(fints,el)
sess = Session()
init(sess)
error()
run(sess, u)


# TODO: change your test parameter to `m`
# gradient check -- v
function scalar_function(m)
    return sum(tanh(fint_comp(fints,el)))
end

# TODO: change `m_` and `v_` to appropriate values
m_ = constant(rand(10,20))
v_ = rand(10,20)
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
