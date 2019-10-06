using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
Random.seed!(233)

################## Load Operator ##################
if Sys.islinux()
py"""
import tensorflow as tf
libSPDOp = tf.load_op_library('build/libSPDOp.so')
@tf.custom_gradient
def spd_op(h0,y):
    out = libSPDOp.spd_op(h0,y)
    def grad(dy):
        return libSPDOp.spd_op_grad(dy, out, h0,y)
    return out, grad
"""
elseif Sys.isapple()
py"""
import tensorflow as tf
libSPDOp = tf.load_op_library('build/libSPDOp.dylib')
@tf.custom_gradient
def spd_op(h0,y):
    out = libSPDOp.spd_op(h0,y)
    def grad(dy):
        return libSPDOp.spd_op_grad(dy, out, h0,y)
    return out, grad
"""
elseif Sys.iswindows()
py"""
import tensorflow as tf
libSPDOp = tf.load_op_library('build/libSPDOp.dll')
@tf.custom_gradient
def spd_op(h0,y):
    out = libSPDOp.spd_op(h0,y)
    def grad(dy):
        return libSPDOp.spd_op_grad(dy, out, h0,y)
    return out, grad
"""
end

spd_op = py"spd_op"
################## End Load Operator ##################

h0 = rand(3,3)
h0 = h0'*h0
ny = 6
y = rand(2, ny)

yi = y[1,:]
L = [yi[1] 0.0 0.0;yi[2] yi[4] 0.0; yi[3] yi[5] yi[6]]
out1 = L * L'
yi = y[2,:]
L =  [yi[1] 0.0 0.0;yi[2] yi[4] 0.0; yi[3] yi[5] yi[6]]
out2 = L * L'

# TODO: specify your input parameters
u = spd_op(constant(h0),constant(y))
sess = Session()
init(sess)
@show run(sess, u)[1,:,:]-out1
@show run(sess, u)[2,:,:]-out2
# error()

# TODO: change your test parameter to `m`
# gradient check -- v
function scalar_function(m)
    return sum(spd_op(constant(h0),m))
end

# TODO: change `m_` and `v_` to appropriate values
m_ = constant(rand(10,ny))
v_ = rand(10,ny)
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

