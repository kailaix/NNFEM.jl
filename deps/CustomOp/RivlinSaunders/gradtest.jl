using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
using NNFEM 
Random.seed!(233)

function rivlin_saunders(strain,c1,c2)
    rivlin_saunders_ = load_op_and_grad("./build/libRivlinSaunders","rivlin_saunders")
    strain,c1,c2 = convert_to_tensor([strain,c1,c2], [Float64,Float64,Float64])
    rivlin_saunders_(strain,c1,c2)
end

c1 = 2.0
c2 = 3.0
N = 100
ε = rand(N,3)
σ = zeros(N,3)
for i = 1:N 
    σ[i,:] = NNFEM.PlaneStressIncompressibleRivlinSaundersStress(ε[i,1],ε[i,2],ε[i,3],c1,c2)
end

# TODO: specify your input parameters
u = rivlin_saunders(ε,c1,c2)
sess = Session(); init(sess)
@show run(sess, u)-σ

# uncomment it for testing gradients
# error() 


# TODO: change your test parameter to `m`
#       in the case of `multiple=true`, you also need to specify which component you are testings
# gradient check -- v
function scalar_function(m)
    # return sum(rivlin_saunders(ε,c1,c2)^2)
    # return sum(rivlin_saunders(m,c1,c2)^2)
    return sum(rivlin_saunders(ε,c1,m)^2)
end

# TODO: change `m_` and `v_` to appropriate values
m_ = constant(rand())
v_ = rand()
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
    w_[i] = s_[i] - g_*sum(v_*dy_)
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
