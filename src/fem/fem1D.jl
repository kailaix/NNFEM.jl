export compute_body_force_terms1, StaticDomain1, compute_stiffness_matrix_and_internal_force1

@doc raw"""
    StaticDomain1(nodes::Array{Float64}, elements::Array,
        EBC::Array{Int64,1}, g::Array{Float64, 1}, FBC::Array{Int64,1}, f::Array{Float64,1},
        edge_traction_data::Array{Int64,2}=zeros(Int64,0,3)))

Constructs a domain for a scalar state variable in a static equation. 

"""
function StaticDomain1(nodes::Array{Float64}, elements::Array,
    EBC::Array{Int64,1}, g::Array{Float64, 1}, FBC::Array{Int64,1}, f::Array{Float64,1},
    edge_traction_data::Array{Int64,2}=zeros(Int64,0,3))
    EBC = reshape(EBC, :, 1)
    FBC = reshape(FBC, :, 1)
    nnodes = size(nodes, 1)
    neles = length(elements)
    ndims = 1

    state = zeros(nnodes * ndims)
    Dstate = zeros(nnodes * ndims)
    LM = Array{Int64}[]
    DOF = Array{Int64}[]
    ID = Int64[]
    neqs = 0
    eq_to_dof = Int64[]
    dof_to_eq = zeros(Bool, nnodes * ndims)
    fext = Float64[]

    npoints = -1
    node_to_point = Int64[]
    history = Dict()

    domain = Domain(nnodes, nodes, neles, elements, ndims, state, Dstate, 
    LM, DOF, ID, neqs, eq_to_dof, dof_to_eq, 
    EBC, g, FBC, fext, edge_traction_data, 0.0, npoints, node_to_point,
    Int64[], Int64[], Int64[], Float64[], 
    Int64[], Int64[], Int64[], Float64[], 
    Int64[], Int64[], Int64[], Float64[], history)

    #set fixed(time-independent) Dirichlet boundary conditions
    setConstantDirichletBoundary!(domain, EBC, g)
    #set constant(time-independent) force load boundary conditions
    setConstantNodalForces!(domain, FBC, f)

    assembleSparseMatrixPattern!(domain)

    domain
end

"""
    compute_stiffness_matrix_and_internal_force1(domain::Domain, Δt::Float64 = 0.0)

Assembles the stiffness matrix and internal force.
"""
function compute_stiffness_matrix_and_internal_force1(domain::Domain, Δt::Float64 = 0.0)
    Fint = zeros(Float64, domain.neqs)
    ii = Int64[]; jj = Int64[]; vv = Float64[]
    neles = domain.neles

    for iele  = 1:neles
        element = domain.elements[iele]

        el_nodes = getNodes(element)

        el_eqns = getEqns(domain, iele)
    
        el_dofs = getDofs(domain, iele)

        el_state  = getState(domain, el_dofs)

        el_Dstate = getDstate(domain, el_dofs)

        fint, stiff  = getStiffAndForce1(element, el_state, el_Dstate, Δt)

        el_eqns_active = el_eqns .>= 1
        Slocal = stiff[el_eqns_active,el_eqns_active]
        Idx = el_eqns[el_eqns_active]
        for i = 1:length(Idx)
            for j = 1:length(Idx)
                push!(ii, Idx[i])
                push!(jj, Idx[j])
                push!(vv, Slocal[i,j])
            end
        end
        Fint[el_eqns[el_eqns_active]] += fint[el_eqns_active]
    end
    Ksparse = sparse(ii, jj, vv, domain.neqs, domain.neqs)

    return Ksparse, Fint
end


@doc raw"""
    compute_body_force_terms1(domain::Domain, f::Function)

Computes the body force 
```math
\int_\Omega f \delta v dx
```
"""
function compute_body_force_terms1(domain::Domain, f::Function)
    
    Fbody = zeros(Float64, domain.neqs)
    neles = domain.neles

    # Loop over the elements in the elementGroup
    for iele  = 1:neles
        element = domain.elements[iele]
  
        gauss_pts = getGaussPoints(element)
        fvalue = f.(gauss_pts[:,1], gauss_pts[:,2])
  
        fbody = getBodyForce(element, fvalue)

      # Assemble in the global array
        el_eqns = getEqns(domain, iele)
        el_eqns_active = (el_eqns .>= 1)
        Fbody[el_eqns[el_eqns_active]] += fbody[el_eqns_active]
    end
  
    return Fbody
end
