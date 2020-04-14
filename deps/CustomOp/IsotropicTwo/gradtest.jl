using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
Random.seed!(233)

function isotropic_two(coef,strain,strainrate)
    isotropic_two_ = load_op_and_grad("./build/libIsotropicTwo","isotropic_two")
    coef,strain,strainrate = convert_to_tensor([coef,strain,strainrate], [Float64,Float64,Float64])
    coef = reshape(coef, (1, 9))
    isotropic_two_(coef,strain,strainrate)
end

γ = rand(9)
N = 1
ε = rand(N, 3)
dotε = rand(N,3)
out = zeros(N,3)
for i = 1:N 
    A = [ε[i,1] ε[i,3]/2
        ε[i,3]/2 ε[i,2]]
    B = [dotε[i,1] dotε[i,3]/2
        dotε[i,3]/2 dotε[i,2]]
    T = γ[1] * I + γ[2] * A + γ[3] * B + γ[4] * A * A + γ[5] * (A*B+B*A) + 
        γ[6] * B * B + γ[7] * (A * A * B + B * A * A) + γ[8] * (A * B * B + B * B * A) + γ[9] * (A * A * B * B + B * B * A * A)
    out[i, :] = [T[1,1];T[2,2];T[1,2]]
end
# TODO: specify your input parameters
u = isotropic_two(γ,ε,dotε)
sess = Session(); init(sess)
@show run(sess, u)-out

# uncomment it for testing gradients
# error() 


# TODO: change your test parameter to `m`
#       in the case of `multiple=true`, you also need to specify which component you are testings
# gradient check -- v
function scalar_function(m)
    # return sum(isotropic_two(γ,ε,dotε)^2)

    return sum(isotropic_two(γ,ε,m)^2)

    # return sum(isotropic_two(m,ε,dotε)^2)

end

# TODO: change `m_` and `v_` to appropriate values
# m_ = constant(rand(9))
# v_ = rand(9)

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
