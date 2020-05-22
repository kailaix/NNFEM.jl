#=
link to 5
conditional realization
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
grad = gradients(loss, θ)|>tf.convert_to_tensor
sess = Session(); init(sess)
tv = matread("data/7.mat")["theta"]
run(sess, loss, θ=>tv)

y = dat[idx]
hs = run(sess, sol[idx])
Hs = matread("data/5.mat")["G"]

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


σv = 0.1
σs = 0.1
R = σv^2*diagm(0=>ones(ndata))
s = tv 
Q = σs^2*diagm(0=>ones(length(θ)))

invR, invQ = inv(R), inv(Q)
Σ = inv(H'*invR*H + invQ)
Σ = (Σ+Σ')/2

Idx = sortperm(s)
d = MvNormal(s[Idx], Σ[Idx, Idx])
cur = rand(d, 50)
# close("all")
# for i = 1:10
#     plot(cur[:,i], color="grey", alpha=0.5)
# end
# plot(s[Idx])

d2 = MvNormal(s, Σ)
est = zeros(length(κ), 200)
@showprogress for i = 1:200
    est[:,i] = run(sess, κ, θ=>rand(d2))
end
M = mean(est, dims=2)[:]
V = std(est, dims=2)[:]

d3 = MvNormal(zeros(length(s)), Q)
est = zeros(length(κ), 200)
@showprogress for i = 1:200
    est[:,i] = run(sess, κ, θ=>rand(d3))
end
M3 = mean(est, dims=2)[:]
V3 = std(est, dims=2)[:]

ii = rand(1:getNGauss(domain),3)
i = ii[3]

close("all")
nm = Normal(M[i], sqrt(V[i]))
x0 = collect(LinRange(1,4,10000))
v = pdf.(nm, x0)
vv = maximum(v)
plot(x0, v)

xy = getGaussPoints(domain)
kk = 2.2 - 0.1*(xy[i,1]^2+xy[i,2]^2)
plot(kk  * ones(100), LinRange(0, vv, 100),"-")

nm = Normal(M3[i], sqrt(V3[i]))
v = pdf.(nm, x0)
plot(x0, v, "--")


# matpcolor(domain, M) #run(sess, κ, θ=>tv))
# matpcolor(domain, abs.(M - run(sess, κ, θ=>tv)))
# matpcolor(domain, V)
