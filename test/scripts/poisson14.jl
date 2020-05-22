#=
Aggregate for testing 

poisson1.jl -- generate data 
poisson9.jl -- train the neural network 
∘ poisson10.jl -- compute the Hessian matrix 
poisson11.jl -- compute posterior of κ
poisson12.jl -- prediction

σv = 0.001 (small noise in dataset)
σs = 0.1

=#

#-----------------------------
# generate data 

using Revise
using NNFEM
using PoreFlow
using ADCME 
using LinearAlgebra
using PyPlot
using MATLAB
using MAT
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
θ = placeholder(matread("data/13_$(σv) .mat")["theta"])
k = squeeze(fc(xy, [20,20,20,1], θ)) + 2.0

k = vector(1:4:4getNGauss(domain), k, 4getNGauss(domain)) + vector(4:4:4getNGauss(domain), k, 4getNGauss(domain))
k = reshape(k, (getNGauss(domain),2,2))
K = s_compute_stiffness_matrix1(k, domain)
S = K\fext

sol = vector(findall(domain.dof_to_eq), S, domain.nnodes)
dat = matread("data/1_dat.mat")["sol"]


idx = sample_interior(domain.nnodes, ndata, bd)

σs = 0.05
loss = sum((sol[idx] - dat[idx])^2)/σv^2 + sum(θ^2)/σs^2 
loss = σv^2 / length(idx) * loss


g = Array{PyObject}(undef, ndata)
for i = 1:ndata
    @info i 
    g[i] = tf.convert_to_tensor(gradients(sol[idx[i]], θ))
end

sess = Session(); init(sess)

@info run(sess, loss)

G = run(sess, g)
matwrite("data/14_$(σv) .mat", Dict(
    "G"=>G
))



