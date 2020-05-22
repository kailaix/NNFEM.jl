#=
Generate data using  
κ = 2.2 - 0.1(x^2+y^2)

same as poisson1.jl, but the source function is different
used for prediction
=#
using Revise
using NNFEM
using PoreFlow
using LinearAlgebra
using PyPlot
using MATLAB
using MAT

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
k = @. 2.2 - 0.1(x^2+y^2)

k = vector(1:4:4getNGauss(domain), k, 4getNGauss(domain)) + vector(4:4:4getNGauss(domain), k, 4getNGauss(domain))
k = reshape(k, (getNGauss(domain),2,2))
K = s_compute_stiffness_matrix1(k, domain)
S = K\fext

sess = Session(); init(sess)
sol[domain.dof_to_eq] = run(sess, S)

@mput nodes sol
matpcolor(domain, sol)
mat"""saveas(gcf, 'figures/1c.png')"""

matwrite("data/1c_dat.mat", Dict(
    "sol"=>sol 
))