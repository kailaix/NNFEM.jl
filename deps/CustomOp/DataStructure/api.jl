using NNFEM



NT = 100
T = 1.0
Δt = T/NT

m, n =  15, 15
h = 1/m
# Create a very simple mesh
elements = []
prop = Dict("name"=> "PlaneStrain", "rho"=> 1.0, "E"=> 2.0, "nu"=> 0.35)
coords = zeros((m+1)*(n+1), 2)
for j = 1:n
    for i = 1:m
        idx = (m+1)*(j-1)+i 
        elnodes = [idx; idx+1; idx+1+m+1; idx+m+1]
        ngp = 3
        nodes = [
            (i-1)*h (j-1)*h
            i*h (j-1)*h
            i*h j*h
            (i-1)*h j*h
        ]
        coords[elnodes, :] = nodes
        push!(elements, SmallStrainContinuum(nodes, elnodes, prop, ngp))
    end
end


# fixed on the bottom, push on the right
EBC = zeros(Int64, (m+1)*(n+1), 2)
FBC = zeros(Int64, (m+1)*(n+1), 2)
g = zeros((m+1)*(n+1), 2)
f = zeros((m+1)*(n+1), 2)

for i = 2:m
    idx = n*(m+1)+i
    EBC[idx,:] .= -2
    EBC[i,:] .= -2
end

for j = 1:n+1 
    idx = (j-1)*(m+1)+1
    EBC[idx,:] .= -2
    idx = (j-1)*(m+1)+m+1
    EBC[idx,:] .= -2
end

ndims = 2
domain = Domain(coords, elements, ndims, EBC, g, FBC, f)


# """
#     init_mesh(domain::Domain)

# Prepares `domain` for use in custom operators.
# """
# function init_mesh(domain::Domain)
#     ccall((:init_mesh, "./build/libdata"), Cvoid, ())

#     for (iele,e) in enumerate(domain.elements)

#         el_eqns = getEqns(domain, iele)
#         el_eqns_active = findall(el_eqns .>= 1)
#         ccall((:create_mesh, "./build/libdata"), Cvoid, 
#         (
#             Ptr{Cint}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Cint,
#             Ptr{Cdouble}, Cint
#         ), e.elnodes[:], e.coords[:], vcat([x[:] for x in e.dhdx]...), e.weights, hcat([x[:] for x in e.hs]...), length(e.elnodes), length(e.weights),
#         el_eqns_active, length(el_eqns_active))
#     end

#     ccall((:create_domain, "./build/libdata"), Cvoid, 
#         (
#             Ptr{Cdouble}, Cint, Cint, Cint
#         ), domain.nodes[:], domain.nnodes, domain.neqs, domain.neles)
#     ccall((:print_mesh, "./build/libdata"), Cvoid, ())
# end


# init_mesh(domain)