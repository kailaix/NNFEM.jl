#=
link to 1c

σv = 0.001 (small noise in dataset)
σs = 0.1

require 1c_dat.mat
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
f = (x,y)->300*sin(2π*y/0.5 + π/8)
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
dat = matread("data/1c_dat.mat")["sol"]


idx = sample_interior(domain.nnodes, ndata, bd)

sess = Session(); init(sess)
tv = matread("data/9.mat")["theta"]
@info run(sess, loss, θ=>tv)


σv = 0.001
σs = 0.05
Σ = matread("data/11.mat")["Sigma"]

d2 = MvNormal(s, Σ)
est = zeros(length(sol), 200)
@showprogress for i = 1:200
    est[:,i] = run(sess, sol, θ=>rand(d2))
end
M = mean(est, dims=2)[:]
V = std(est, dims=2)[:]

d3 = MvNormal(zeros(length(s)), Q)
est3 = zeros(length(sol), 200)
@showprogress for i = 1:200
    est3[:,i] = run(sess, sol, θ=>rand(d3))
end
M3 = mean(est3, dims=2)[:]
V3 = std(est3, dims=2)[:]

Random.seed!(2333)
idx = rand(1:domain.nnodes, 3)
for k = 1:3
    i = idx[k]
    @info i 

    close("all")
    # x0 = collect(LinRange(1.7,2.3,10000))
    
    # nm = Normal(M3[i], V3[i])
    # v = pdf.(nm, x0)
    # plot(x0, v, "--")
    hist(est3[i,:], bins=20, alpha=0.8, density=true, label="Prior")
    
    
    # nm = Normal(M[i], V[i])
    # v = pdf.(nm, x0)
    hist(est[i,:], bins=20, density=true, alpha=0.8, label="Posterior")
    # vv = maximum(v)
    # plot(x0, v, "--")
    
    # xy = getGaussPoints(domain)
    # kk = 2.2 - 0.1*(xy[i,1]^2+xy[i,2]^2)
    plot(dat[i]  * ones(100), LinRange(0, 20, 100), "k-", linewidth=2, label="Reference")
    
    legend()
    
    xlabel("\$u(x, y)\$")
    ylabel("Density")
    
    
    savefig("figures/11_$k.png")
end
# matpcolor(domain, M)
# matpcolor(domain, V)

# matpcolor(domain, M3)
# matpcolor(domain, V3)
