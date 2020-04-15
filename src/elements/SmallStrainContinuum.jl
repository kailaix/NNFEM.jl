include("Continuum.jl")
export SmallStrainContinuum 

@doc raw"""
    SmallStrainContinuum

Constructs a small strain element. 

- `eledim`: spatial dimension of the element (default = 2).
- `mat`: constitutive law, a length `#elem` vector of materials such as [`PlaneStress`](@ref)
- `elnodes`: the node indices in this finite element, an integer array 
- `props`: property dictionary 
- `coords`: coordinates of the vertices of the element
- `dhdx`: list of `ngp` shape functions for first order derivatives $\nabla \phi(x)$ (`ndof×2`) on the Gaussian points
- `weights`: weight vector of length `n_gauss_points`, for numerical quadrature
- `hs`: list of `ngp` shape functions for function values $\phi(x)$ (length `ndof` vectors) on the Gaussian points
- `stress`: stress at each quadrature points; this field is reserved for visualization. 

# Example
```julia
#   Local degrees of freedom 
#   4 ---- 3
#
#   1 ---- 2

nx = 10
ny = 5
h = 0.1
element = SmallStrainContinuum[]
prop = Dict("name"=> "PlaneStrain", "rho"=> 0.0876584, "E"=>0.07180760098, "nu"=>0.4)
for j = 1:ny
    for i = 1:nx 
        n = (nx+1)*(j-1) + (i-1)+1
        elnodes = [n, n + 1, n + 1 + (nx + 1), n + (nx + 1)]
        ngp = 3 # 3 x 3 Gauss points per element 
        coords = [(i-1)*h (j-1)*h
                    i*h (j-1)*h
                    i*h j*h
                    (i-1)*h j*h]
        push!(element, SmallStrainContinuum(coords,elnodes, prop, ngp))
    end
end
```

"""
mutable struct SmallStrainContinuum <: Continuum 
    eledim::Int64
    mat  # constitutive law
    elnodes::Array{Int64}   # the node indices in this finite element
    props::Dict{String, Any}
    coords::Array{Float64}
    dhdx::Array{Array{Float64}}  # 4nPointsx2 matrix
    weights::Array{Float64} 
    hs::Array{Array{Float64}}
    stress::Array{Array{Float64}}
end

Base.show(io::IO, z::SmallStrainContinuum) = print(io, "SmallStrainContinuum with $(length(z.weights)) Gauss quadrature points and $(z.props["name"]) material.")

"""
    SmallStrainContinuum(coords::Array{Float64}, elnodes::Array{Int64}, props::Dict{String, Any}, ngp::Int64=2)
"""
function SmallStrainContinuum(coords::Array{Float64}, elnodes::Array{Int64}, props::Dict{String, Any}, ngp::Int64=2)
    eledim = 2
    # @show coords, ngp

    dhdx, weights, hs = get2DElemShapeData( coords, ngp )
    nGauss = length(weights)
    name = props["name"]
    if name=="PlaneStrain"
        mat = [PlaneStrain(props) for i = 1:nGauss]
    elseif name=="PlaneStress"
        mat = [PlaneStress(props) for i = 1:nGauss]
    elseif name=="PlaneStressPlasticityLawBased"
        mat = [PlaneStressPlasticityLawBased(props) for i = 1:nGauss]
    elseif name=="PlaneStressPlasticity"
        mat = [PlaneStressPlasticity(props) for i = 1:nGauss]
    elseif name=="NeuralNetwork2D"
        mat = [NeuralNetwork2D(props) for i = 1:nGauss]
    elseif name=="PlaneStressInpressibleRivlinSaunders"
        mat = [PlaneStressIncompressibleRivlinSaunders(props) for i = 1:nGauss]
    else
        error("Not implemented yet: $name")
    end
    strain = Array{Array{Float64}}(undef, length(weights))
    SmallStrainContinuum(eledim, mat, elnodes, props, coords, dhdx, weights, hs, strain)
end

"""
    getStiffAndForce(elem::SmallStrainContinuum, state::Array{Float64}, Dstate::Array{Float64}, Δt::Float64)

Returns the internal force term and the stiffness matrix. `state` and `Dstate` are restriction of full state variables to this element. 
"""
function getStiffAndForce(elem::SmallStrainContinuum, state::Array{Float64}, Dstate::Array{Float64}, Δt::Float64)
    ndofs = dofCount(elem); 
    nnodes = length(elem.elnodes)
    fint = zeros(Float64, ndofs)
    stiff = zeros(Float64, ndofs,ndofs)
    out = Array{Float64}[]
    u = state[1:nnodes]; v = state[nnodes+1:2*nnodes]
    Du = Dstate[1:nnodes]; Dv = Dstate[nnodes+1:2*nnodes]
    # #@show "u ", u, " Du ", Du
    # #@show "v " v, " Dv " Dv

    for k = 1:length(elem.weights)
        # #@show "Gaussian point ", k
        g1 = elem.dhdx[k][:,1]; g2 = elem.dhdx[k][:,2]
        
        ux = u'*g1; uy = u'*g2; vx = v'*g1; vy = v'*g2
        Dux = Du'*g1; Duy = Du'*g2; Dvx = Dv'*g1; Dvy = Dv'*g2
        #@show "gauss", k , u, Du, ux, Dux
        
        # compute  ∂E∂u.T, 8 by 3 array 
        ∂E∂u = [g1   zeros(nnodes)    g2;
                zeros(nnodes)    g2   g1;] 
        
        E = [ux; vy; uy+vx]
        DE = [Dux; Dvy; Duy+Dvx]

        

        # #@show E, DE
        S, dS_dE = getStress(elem.mat[k], E, DE, Δt)

        # @info "gauss ", k, " E ", E, " S ", S

        elem.stress[k] = S
        # @show size(S), size(∂E∂u)
        fint += ∂E∂u * S * elem.weights[k] # 1x8
        
        stiff += (∂E∂u * dS_dE * ∂E∂u')*elem.weights[k] # 8x8

    end
    
    return fint, stiff
end

"""
    getInternalForce(elem::SmallStrainContinuum, state::Array{Float64}, Dstate::Array{Float64}, Δt::Float64)

Returns the internal force term. `state` and `Dstate` are restriction of full state variables to this element. 
"""
function getInternalForce(elem::SmallStrainContinuum, state::Array{Float64}, Dstate::Array{Float64}, Δt::Float64)
    n = dofCount(elem)
    nnodes = length(elem.elnodes)
    fint = zeros(Float64,n)
    out = Array{Float64}[]
    u = state[1:nnodes]; v = state[nnodes+1:end]
    Du = Dstate[1:nnodes]; Dv = Dstate[nnodes+1:end]
    for k = 1:length(elem.weights)
        g1 = elem.dhdx[k][:,1]; g2 = elem.dhdx[k][:,2]
        
        ux = u'*g1; uy = u'*g2; vx = v'*g1; vy = v'*g2
        Dux = Du'*g1; Duy = Du'*g2; Dvx = Dv'*g1; Dvy = Dv'*g2
        # compute  ∂E∂u.T, 8 by 3 array 
        ∂E∂u = [g1   zeros(nnodes)    g2;
                zeros(nnodes)    g2   g1;] 
        
        E = [ux; vy; uy+vx]
        DE = [Dux; Dvy; Duy+Dvx]

        S, dS_dE = getStress(elem.mat[k], E, DE, Δt)

        elem.stress[k] = S

        fint += ∂E∂u * S * elem.weights[k] # 1x8
    end
    return fint
end



"""
    getStrain(elem::SmallStrainContinuum, state::Array{Float64})

Returns the strain of this element.  `state` is restricted to this variable. 
"""
function getStrain(elem::SmallStrainContinuum, state::Array{Float64})
    n = dofCount(elem)
    nnodes = length(elem.elnodes)
    u = state[1:nnodes]; v = state[nnodes+1:end]
    nGauss = length(elem.weights)
    E = zeros(nGauss, 3)
    w∂E∂u = zeros(nGauss, 2nnodes, 3)
    for k = 1:nGauss
        g1 = elem.dhdx[k][:,1]; g2 = elem.dhdx[k][:,2]
        ux = u'*g1; uy = u'*g2; vx = v'*g1; vy = v'*g2
        # compute  ∂E∂u.T, 8 by 3 array 
        w∂E∂u[k,:,:] = [g1   zeros(nnodes)    g2;
                zeros(nnodes)    g2   g1;] * elem.weights[k]
        E[k,:] = [ux; vy; uy+vx]
    end
    return E, w∂E∂u
end

# for the adjoint solver


# Return: 
#    strain{ngps_per_elem, nstrain} 
#    dstrain_dstate_tran{neqs_per_elem, ngps_per_elem*nstrain}  
function getStrainState(elem::SmallStrainContinuum, state::Array{Float64})
    n = dofCount(elem)
    nnodes = length(elem.elnodes)
    u = state[1:nnodes]; v = state[nnodes+1:end]
    nGauss = length(elem.weights)
    nStrain = 3
    E = zeros(nGauss, nStrain)
    ∂E∂u = zeros(nGauss * nStrain, 2nnodes)

    for k = 1:nGauss
        g1 = elem.dhdx[k][:,1]; g2 = elem.dhdx[k][:,2]
        ux = u'*g1; uy = u'*g2; vx = v'*g1; vy = v'*g2
        E[k,:] = [ux; vy; uy+vx]

        # compute  ∂E∂u
        ∂E∂u[(k-1)*nStrain+1:k*nStrain, :] = 
                [g1   zeros(nnodes)    g2;
                zeros(nnodes)    g2   g1;]'
        
    end
    return E, ∂E∂u
end


function getStiffAndForce(elem::SmallStrainContinuum, state::Array{Float64},
                          stress::Array{Float64,2}, dstress_dstrain_T::Array{Float64,3})
    ndofs = dofCount(elem); 
    nnodes = length(elem.elnodes)
    fint = zeros(Float64, ndofs)
    stiff = zeros(Float64, ndofs,ndofs)

    u = state[1:nnodes]; v = state[nnodes+1:2*nnodes]

    for k = 1:length(elem.weights)
        g1 = elem.dhdx[k][:,1]; g2 = elem.dhdx[k][:,2]
        
        # compute  ∂E∂u.T, 8 by 3 array 
        ∂E∂u = [g1   zeros(nnodes)    g2;
                zeros(nnodes)    g2   g1;]  

        S, dS_dE_T = stress[k, :], dstress_dstrain_T[k,:,:]


        elem.stress[k] = S
        # @show size(S), size(∂E∂u)
        fint += ∂E∂u * S * elem.weights[k] # 1x8
        
        stiff += (∂E∂u * dS_dE_T' * ∂E∂u')*elem.weights[k] # 8x8
    end
    return fint, stiff
end

function  getStiffAndDforceDstress(elem::SmallStrainContinuum, state::Array{Float64},  
    stress::Array{Float64,2}, dstress_dstrain_T::Array{Float64,3})
    ndofs = dofCount(elem); 
    nnodes = length(elem.elnodes)
    nStrain = 3
    nGauss = length(elem.weights)
    stiff = zeros(Float64, ndofs,ndofs)
    dfint_dstress = zeros(Float64,  ndofs, nGauss * nStrain)

    u = state[1:nnodes]; v = state[nnodes+1:2*nnodes]

    for k = 1:length(elem.weights)
        g1 = elem.dhdx[k][:,1]; g2 = elem.dhdx[k][:,2]

        
        # compute  ∂E∂u, 3 by 2nnodes array 
        ∂E∂u = [g1   zeros(nnodes)    g2;
                zeros(nnodes)    g2   g1;]

        # #@show E, DE
        S, dS_dE_T = stress[k, :], dstress_dstrain_T[k,:,:]

        elem.stress[k] = S

        #fint += ∂E∂u * S * elem.weights[k] # 1x8
        
        dfint_dstress[:, (k-1)*nStrain+1:k*nStrain] = ∂E∂u * elem.weights[k]

        stiff += (∂E∂u * dS_dE_T' * ∂E∂u')*elem.weights[k] # 8x8
    end
    return stiff , dfint_dstress
end



function  getForceAndDforceDstress(elem::SmallStrainContinuum, state::Array{Float64},  
    stress::Array{Float64,2})
    ndofs = dofCount(elem); 
    nnodes = length(elem.elnodes)
    nStrain = 3
    nGauss = length(elem.weights)
    fint = zeros(Float64, ndofs)
    dfint_dstress = zeros(Float64,  ndofs, nGauss * nStrain)

    u = state[1:nnodes]; v = state[nnodes+1:2*nnodes]

    for k = 1:length(elem.weights)
        g1 = elem.dhdx[k][:,1]; g2 = elem.dhdx[k][:,2]

        # compute  ∂E∂u, 3 by 2nnodes array 
        ∂E∂u = [g1   zeros(nnodes)    g2;
                zeros(nnodes)    g2   g1;]

        # #@show E, DE
        S = stress[k, :]

        elem.stress[k] = S

        fint += ∂E∂u * S * elem.weights[k] # 1x8
        
        dfint_dstress[:, (k-1)*nStrain+1:k*nStrain] = ∂E∂u * elem.weights[k]

    end
    return fint , dfint_dstress
end
