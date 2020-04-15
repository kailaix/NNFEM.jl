using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
Random.seed!(233)

function tensor_rep(inp)
    tensor_rep_ = load_op_and_grad("./build/libTensorRep","tensor_rep")
    inp = convert_to_tensor([inp], [Float64]); inp = inp[1]
    tensor_rep_(inp)
end

# TODO: specify your input parameters
N = 100
inp = rand(N,3)
out = zeros(N,2,2)
for i = 1:N 
    out[i,:,:] = [inp[i, 1] inp[i,3]/2
    inp[i,3]/2 inp[i,2]]
end
u = tensor_rep(inp)
sess = Session(); init(sess)
@show run(sess, u)-out

# uncomment it for testing gradients
# error() 


# TODO: change your test parameter to `m`
#       in the case of `multiple=true`, you also need to specify which component you are testings
# gradient check -- v
function scalar_function(m)
    return sum(tensor_rep(m)^2)
end

# TODO: change `m_` and `v_` to appropriate values
m_ = constant(rand(N,3))
v_ = rand(N,3)
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
