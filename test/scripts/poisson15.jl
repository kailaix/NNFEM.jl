#=
Aggregate for testing 

poisson1.jl -- generate data 
poisson9.jl -- train the neural network 
poisson10.jl -- compute the Hessian matrix 
∘ poisson11.jl -- compute posterior of κ
poisson12.jl -- prediction

σv = 0.001 (small noise in dataset)
σs = 0.1

save best neural network to data13
=#

using Revise
using NNFEM
using PoreFlow
using ADCME 
using LinearAlgebra
using PyPlot
using MATLAB
using PyCall
using Distributions
using ProgressMeter
using MAT
using Statistics
include("common1.jl")

ndata = 20
nodes, elems = meshread("$(splitdir(pathof(NNFEM))[1])/../deps/Data/lshape.msh")
elements = []
prop = Dict("name"=> "Scalar1D", "kappa"=>2.0)

for k = 1:size(elems,1)
    elnodes = elems[k,:]
    ngp = 2
    coord = nodes[elnodes,:]
    push!(elements, SmallStrainContinuum(coord, elnodes, prop, ngp))
end


# free boundary on all sides
EBC = zeros(Int64, size(nodes,1))
FBC = zeros(Int64, size(nodes,1))
g = zeros(size(nodes,1))
f = zeros(size(nodes,1))

bd = find_boundary(nodes, elems)
EBC[bd] .= -1

ndims = 1
domain = StaticDomain1(nodes, elements, EBC, g, FBC, f)
init_nnfem(domain)

α = 0.4*π/2
d = [cos(α);sin(α)]
f = (x,y)->300*sin(2π*y + π/8)
fext = compute_body_force_terms1(domain, f)

sol = zeros(domain.nnodes)
xy = getGaussPoints(domain)
x = xy[:,1]
y = xy[:,2]
θ = placeholder(fc_init([2,20,20,20,1]))
κ = squeeze(fc(xy, [20,20,20,1], θ)) + 2.0

k = vector(1:4:4getNGauss(domain), κ, 4getNGauss(domain)) + vector(4:4:4getNGauss(domain), κ, 4getNGauss(domain))
k = reshape(k, (getNGauss(domain),2,2))
K = s_compute_stiffness_matrix1(k, domain)
S = K\fext

sol = vector(findall(domain.dof_to_eq), S, domain.nnodes)
dat = matread("data/1_dat.mat")["sol"]


idx = sample_interior(domain.nnodes, ndata, bd)

loss = mean((sol[idx] - dat[idx])^2)
sess = Session(); init(sess)
tv = matread("data/13_$(σv).mat")["theta"]
@info run(sess, loss, θ=>tv)
y = dat[idx]
hs = run(sess, sol[idx])
Hs = matread("data/14_$(σv).mat")["G"]

H = zeros(ndata, length(θ))
for i = 1:ndata
    h = Hs[i,:]
    o = 0
    for j = 1:length(h)
        m = length(h[j])
        H[i, o+1:o+m] = typeof(h[j])<:Real ? [h[j]] : h[j]'[:]
        o = o + m 
    end
end


σs = 0.05
R = σv^2*diagm(0=>ones(ndata))
s = tv 
Q = σs^2*diagm(0=>ones(length(θ)))

invR, invQ = inv(R), inv(Q)
Σ = inv(H'*invR*H + invQ)
Σ = (Σ+Σ')/2

matwrite("data/15_$(σv).mat", Dict(
    "Sigma"=>Σ
))
Idx = sortperm(s)
d = MvNormal(s[Idx], Σ[Idx, Idx])
cur = rand(d, 50)
# close("all")
# for i = 1:10
#     plot(cur[:,i], color="grey", alpha=0.5)
# end
# plot(s[Idx])

d2 = MvNormal(s, Σ)
est = zeros(length(κ), 1000)
@showprogress for i = 1:1000
    est[:,i] = run(sess, κ, θ=>rand(d2))
end
M = mean(est, dims=2)[:]
V = std(est, dims=2)[:]

d3 = MvNormal(zeros(length(s)), Q)
est3 = zeros(length(κ), 1000)
@showprogress for i = 1:1000
    est3[:,i] = run(sess, κ, θ=>rand(d3))
end
M3 = mean(est3, dims=2)[:]
V3 = std(est3, dims=2)[:]

Random.seed!(2333)
ii = rand(1:getNGauss(domain),3)

for k = 1:3
    i = ii[k]
@info i 
close("all")
x0 = collect(LinRange(1.7,2.3,10000))

nm = Normal(M3[i], V3[i])
v = pdf.(nm, x0)
plot(x0, v, "--")
hist(est3[i,:], bins=20, alpha=0.8, density=true, label="Prior")


nm = Normal(M[i], V[i])
v = pdf.(nm, x0)
hist(est[i,:], bins=20, density=true, alpha=0.8, label="Posterior")
vv = maximum(v)
plot(x0, v, "--")

xy = getGaussPoints(domain)
kk = 2.2 - 0.1*(xy[i,1]^2+xy[i,2]^2)
plot(kk  * ones(100), LinRange(0, 1.1*vv, 100), "k-", linewidth=2, label="Reference")

legend()

xlabel("\$\\kappa(x)\$")
ylabel("Density")


savefig("figures/15_$(σv)_$k.png")

end


# matpcolor(domain, V)
# matpcolor(domain, M)
# # prior  
# matpcolor(domain, M3)
# matpcolor(domain, V3)
