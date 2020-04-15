using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
Random.seed!(233)

function isotropic(coef,strain)
    isotropic_ = load_op_and_grad("./build/libIsotropic","isotropic")
    coef,strain = convert_to_tensor([coef,strain], [Float64,Float64])
    isotropic_(coef,strain)
end

# TODO: specify your input parameters
N = 10
strain = rand(N,3)
coef = rand(N, 3)

out = zeros(N,3)
for i = 1:N 
    A = [strain[i,1] strain[i,3]/2
        strain[i,3]/2 strain[i,2]]
    S = coef[i,1]*I + coef[i,2] * A + coef[i,3] * A * A
    out[i,:] = [S[1,1];S[2,2];S[1,2]]
end
u = isotropic(coef,strain)
sess = Session(); init(sess)
@show run(sess, u)-out

# uncomment it for testing gradients
# error() 


# TODO: change your test parameter to `m`
#       in the case of `multiple=true`, you also need to specify which component you are testings
# gradient check -- v
function scalar_function(m)
    # return sum(isotropic(coef,strain)^2)
    return sum(isotropic(coef,m)^2)
end

# TODO: change `m_` and `v_` to appropriate values
# m_ = constant([ zeros(N,1) rand(N,1) zeros(N,1)])
# v_ = [ zeros(N,1) rand(N,1) zeros(N,1)]

m_ = constant([ zeros(N,2) rand(N,1)])
v_ = [ zeros(N,2) rand(N,1)]
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
