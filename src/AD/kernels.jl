
export init_nnfem, example_domain, 
s_eval_strain_on_gauss_points, s_compute_stiffness_matrix,
s_compute_internal_force_term, example_global_data
"""
    init_nnfem(domain::Domain)

Prepares `domain` for use in custom operators.
"""
function init_nnfem(domain::Domain)
    LIB = joinpath(@__DIR__, "../../deps/CustomOp/DataStructure/build/libdata")

    @eval begin 
        ccall((:init_mesh, $LIB), Cvoid, ())

        for (iele,e) in enumerate($domain.elements)

            el_eqns = getEqns($domain, iele)
            el_eqns_active = findall(el_eqns .>= 1)
            ccall((:create_mesh, $LIB), Cvoid, 
            (
                Ptr{Cint}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Cint,
                Ptr{Cdouble}, Cint, Ptr{Cint}
            ), Int32.(e.elnodes[:]), e.coords[:], vcat([x[:] for x in e.dhdx]...), e.weights, hcat([x[:] for x in e.hs]...), Int32(length(e.elnodes)), Int32(length(e.weights)),
            Int32.(el_eqns_active), Int32(length(el_eqns_active)), Int32.(el_eqns))
        end

        ccall((:create_domain, $LIB), Cvoid, 
            (
                Ptr{Cdouble}, Cint, Cint, Cint
            ), $domain.nodes[:], Int32.($domain.nnodes), Int32.($domain.neqs), Int32.($domain.neles))
        ccall((:print_mesh, $LIB), Cvoid, ())

    end
end


function example_domain(m::Int64 = 10, n::Int64 = 10, h::Float64 = 1.0)
    # Create a very simple mesh
    elements = []
    prop = Dict("name"=> "PlaneStrain", "rho"=> 1.0, "E"=> 2.0, "nu"=> 0.35)
    coords = zeros((m+1)*(n+1), 2)
    for j = 1:n
        for i = 1:m
            idx = (m+1)*(j-1)+i 
            elnodes = [idx; idx+1; idx+1+m+1; idx+m+1]
            ngp = 2
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
        FBC[idx,:] .= -1
        FBC[i,:] .= -1
    end

    for j = 1:n+1 
        idx = (j-1)*(m+1)+1
        FBC[idx,:] .= -1
        idx = (j-1)*(m+1)+m+1
        FBC[idx,:] .= -1
    end

    ndims = 2
    domain = Domain(coords, elements, ndims, EBC, g, FBC, f)
end

function example_global_data(domain::Domain)
    Dstate = zeros(domain.neqs)
    state = zeros(domain.neqs)
    velo = zeros(domain.neqs)
    acce = zeros(domain.neqs)
    gt = nothing
    ft = nothing
    globdat = GlobalData(state, Dstate, velo, acce, domain.neqs, gt, ft)
end


@doc raw"""
    s_compute_stiffness_matrix(k::Union{Array{Float64,3}, PyObject})

Computes the small strain stiffness matrix. $k$ is a $n\times 3\times 3$ matrix, where $n$ is the total number of Gauss points.
Returns a SparseTensor. 
"""
function s_compute_stiffness_matrix(k::Union{Array{Float64,3}, PyObject}, domain::Domain)
    small_continuum_stiffness_ = load_op_and_grad("$(@__DIR__)/../../deps/CustomOp/SmallContinuumStiffness/build/libSmallContinuumStiffness","small_continuum_stiffness", multiple=true)
    k = convert_to_tensor([k], [Float64]); k = k[1]
    ii, jj, vv = small_continuum_stiffness_(k)
    SparseTensor(ii+1, jj+1, vv, domain.neqs, domain.neqs)
end


@doc raw"""
    s_eval_strain_on_gauss_points(state::Union{Array{Float64,1}, PyObject})

Computes the strain on Gauss points in the small strain case. `state` is the full displacement vector. 
"""
function s_eval_strain_on_gauss_points(state::Union{Array{Float64,1}, PyObject}, domain::Domain)
    small_continuum_strain_ = load_op_and_grad("$(@__DIR__)/../../deps/CustomOp/SmallContinuumStrain/build/libSmallContinuumStrain","small_continuum_strain")
    state = convert_to_tensor([state], [Float64]); state = state[1]
    ep = small_continuum_strain_(state)
    set_shape(ep, (getNGauss(domain), 3))
end


@doc raw"""
    s_compute_internal_force_term(stress::Union{Array{Float64,2}, PyObject})

Computes the internal force
$$\int_\Omega \sigma : \delta \epsilon dx$$
Only active DOFs are considered. 
"""
function s_compute_internal_force_term(stress::Union{Array{Float64,2}, PyObject}, domain::Domain)
    small_continuum_fint_ = load_op_and_grad("$(@__DIR__)/../../deps/CustomOp/SmallContinuumFint/build/libSmallContinuumFint","small_continuum_fint")
    stress = convert_to_tensor([stress], [Float64]); stress = stress[1]
    out = small_continuum_fint_(stress)
    set_shape(out, (domain.neqs,))
end