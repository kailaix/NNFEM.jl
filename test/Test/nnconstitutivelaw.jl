using NNFEM
using PyPlot
n = 1
p = 9*20+20+20*20+20+20*3+3
θ = 0.01*rand(p)
input = 0.1*rand(n,9)
g = rand(n,3)

# v = rand(683)
# A0, B0, C0 = nn_constitutive_law(input, θ, g, 0, 1)
# γs = [1.,1e-1,1e-2,1e-3,1e-4]
# L = zeros(5); Δ = zeros(5); δ = zeros(5)
# for i = 1:5
#     A, B, C = nn_constitutive_law(input, γs[i]*v + θ, g, 0, 1)
#     L[i] = sum(A.*g)
#     Δ[i] = L[i] - sum(A0.*g)
#     δ[i] = L[i] - sum(A0.*g) - γs[i]*sum(v.*C0)
# end

@assert n==1
v = rand(1,9)
A0, B0, C0 = nn_constitutive_law(input, θ, nothing, 1, 0)
g = rand(1, 3)
γs = [1.,1e-1,1e-2,1e-3,1e-4]
L = zeros(5); Δ = zeros(5); δ = zeros(5)
for i = 1:5
    A, B, C = nn_constitutive_law(γs[i]*v +input,  θ)
    L[i] = sum(A.*g)
    Δ[i] = L[i] - sum(A0.*g)
    δ[i] = L[i] - sum(A0.*g) - γs[i]*sum(v[:].*(B0[1,:,:]*g[:]))
end

sval_ = Δ
wval_ = δ
gs_ = γs
close("all")
loglog(gs_, abs.(sval_), "*-", label="finite difference")
loglog(gs_, abs.(wval_), "+-", label="automatic differentiation")
loglog(gs_, gs_.^2 * 0.5*abs(wval_[1])/gs_[1]^2, "--",label="\$\\mathcal{O}(\\gamma^2)\$")
loglog(gs_, gs_ * 0.5*abs(sval_[1])/gs_[1], "--",label="\$\\mathcal{O}(\\gamma)\$")

plt.gca().invert_xaxis()
legend()
xlabel("\$\\gamma\$")
ylabel("Error")

