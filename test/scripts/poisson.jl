using Revise
using NNFEM
using PoreFlow
using LinearAlgebra
using PyPlot
using MATLAB

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

α = 0.4*π/2
d = [cos(α);sin(α)]
f = (x,y)->sin(2π*y + π/8)
fext = compute_body_force_terms1(domain, f)

sol = zeros(domain.nnodes)
K, fint = compute_stiffness_matrix_and_internal_force1(domain)
sol[domain.dof_to_eq] = K\fext

@mput nodes sol
mat"""
x = nodes(:,1)
y = nodes(:,2)
tri = delaunay(x,y);
plot(x,y,'.')
h = trisurf(tri, x, y, sol)
"""

