export example_domain, init_nnfem, example_global_data, example_static_domain1, find_boundary

@doc raw"""
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


function example_static_domain1(m::Int64 = 10, n::Int64 = 10, h::Float64 = 1.0)
    # Create a very simple mesh
    elements = []
    prop = Dict("name"=> "Scalar1D", "kappa"=>1.0 )
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
    
    
    # free boundary on all sides
    EBC = zeros(Int64, (m+1)*(n+1))
    FBC = zeros(Int64, (m+1)*(n+1))
    g = zeros((m+1)*(n+1))
    f = zeros((m+1)*(n+1))
    
    for j = 1:n+1 
        FBC[(j-1)*(m+1) + 1] = -1
        FBC[(j-1)*(m+1) + m + 1] = -1
    end

    for i = 1:m+1 
        FBC[  i ] = -1
        FBC[ n*(m+1) + i ] = -1
    end
    
    ndims = 1
    domain = StaticDomain1(coords, elements, EBC, g, FBC, f)
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

function find_boundary(nodes::Array{Float64, 2}, elements::Array{Int64, 2})
    ef = (i, j)->(min(i, j), max(i, j))
    push_pop! = x -> (x in eset ? pop!(eset, x) : push!(eset, x))
    eset = Set([])
    for k = 1:size(elements, 1)
        push_pop!(ef(elements[k,1], elements[k,2]))
        push_pop!(ef(elements[k,2], elements[k,3]))
        push_pop!(ef(elements[k,3], elements[k,4]))
        push_pop!(ef(elements[k,4], elements[k,1]))
    end
    nset = Set(Int64[])
    for e in collect(eset)
        push!(nset, e[1])
        push!(nset, e[2])
    end
    return collect(nset)
end