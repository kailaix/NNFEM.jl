export getStiffAndForce, getInternalForce,
getBodyForce, getEdgeForce, getStrain, getMassMatrix,
getNodes, getGaussPoints, getEdgeGaussPoints, commitHistory

abstract type Continuum end 

@doc raw"""
    getBodyForce(elem::Continuum, fvalue::Array{Float64,2})

Returns the body force. 

- `fvalue` is a $n_{gauss}\times 2$ matrix, which is ordered the same as 
    Gaussian points in the undeformed parent element.

Returns the nodal force due to the body force
$$\int_{e} \mathbf{f}(\mathbf{x})\cdot \delta \mathbf{u}(\mathbf{x}) d \mathbf{x} 
  = \int_{e} \mathbf{f}(\mathbf{\xi})\cdot \delta \mathbf{u}(\mathbf{\xi}) 
  |\frac{\partial \mathbf{x}}{\partial \mathbf{\xi}}| d \mathbf{\xi}$$

!!! todo 
    Add force in the deformed domain.
"""
function getBodyForce(elem::Continuum, fvalue::Array{Float64,2})
    n = dofCount(elem)
    fbody = zeros(Float64,n)

    nnodes = length(elem.elnodes)
    for k = 1:length(elem.weights)
        fbody[1:nnodes] += elem.hs[k] * fvalue[k,1] * elem.weights[k]
        fbody[nnodes+1:2*nnodes] += elem.hs[k] * fvalue[k,2] * elem.weights[k]
    end
    return fbody
end

@doc raw"""
    getBodyForce(elem::Continuum, fvalue::Array{Float64, 1})

Returns 
```math 
\int_A f \delta v dx 
```
on a specific element $A$
`fvalue` has the same length as number of Gauss points. 
"""
function getBodyForce(elem::Continuum, fvalue::Array{Float64, 1})
    nnodes = length(elem.elnodes)
    fbody = zeros(nnodes)
    for k = 1:length(elem.weights)
        fbody += elem.hs[k] * fvalue[k] * elem.weights[k]
    end
    return fbody
end

@doc raw"""
    getEdgeForce(elem::Continuum, iedge::Float64, fvalue::Array{Float64,2})
    
Returns the force imposed by boundary tractions.

`fvalue` is a $n_{edge_gauss}\times 2$ matrix, which is ordered the same as the
    Gaussian points in undeformed parent edge element.
    The element nodes are ordered as 
    #   4 ---- 3             #   4 --7-- 3
    #                        #   8   9   6 
    #   1 ---- 2             #   1 --5-- 2
    for porder=1     or          porder=2
    iedge 1, 2, 3, 4 are (1,2), (2,3), (3,4), (4,1)
                    are (1,2,5), (2,3,6), (3,4,7), (4,1,8)

Returns the nodal force due to the traction on the iedge-th edge of the element
$$\int_{s} \mathbf{f}(\mathbf{x})\cdot \delta \mathbf{u}(\mathbf{x}) d s 
  = \int_{e} \mathbf{f}(\xi)\cdot \delta \mathbf{u}(\xi) 
  |\frac{\partial \mathbf{x}}{\partial \xi}| d \xi$$

!!! todo 
    This function imposes force in the undeformed domain. Add force in the deformed domain in the future.
"""
function getEdgeForce(elem::Continuum, iedge::Int64, fvalue::Array{Float64,2})
    n = length(elem.elnodes)
    ngp = Int64(sqrt(length(elem.weights)))
    @assert(n == 4 || n == 9)

    n1, n2 = iedge, ((iedge+1)==5 ? 1 : iedge+1)
    loc_id = (n == 4 ? [n1, n2] : [n1, n2, iedge+4])

    x = elem.coords[loc_id,:]

    weights, hs = get1DElemShapeData(x, ngp)  

    fedge = zeros(Float64,2n)
    for igp = 1:ngp
        fedge[loc_id] += hs[igp] * fvalue[igp,1] * weights[igp]
        fedge[n .+ loc_id] += hs[igp] * fvalue[igp,2] * weights[igp]
    end
    return fedge


end

"""
    getMassMatrix(elem::Continuum)

Returns the mass matrix and lumped mass matrix of the element `elem`.
"""
function getMassMatrix(elem::Continuum)
    ndofs = dofCount(elem)
    nnodes = length(elem.elnodes)
    mass = zeros(ndofs,ndofs)
    for k = 1:length(elem.weights)
        rho = elem.mat[k].ρ
        mass += [elem.hs[k]*elem.hs[k]' zeros(nnodes, nnodes)
                 zeros(nnodes, nnodes)  elem.hs[k]*elem.hs[k]']  * rho * elem.weights[k]
    end
    lumped = sum(mass, dims=2)
    mass, lumped
end

""" 
    getNodes(elem::Continuum)

Alias for `elem.elnodes`
"""
function getNodes(elem::Continuum)
    return elem.elnodes
end

"""
    getGaussPoints(elem::Continuum)

Returns the Gauss quadrature nodes of the element in the undeformed domain
"""
function getGaussPoints(elem::Continuum)
    x = elem.coords'
    gnodes = zeros(length(elem.weights),2)
    for k = 1:length(elem.weights)
        gnodes[k,:] = x * elem.hs[k] 
    end
    return gnodes
end

"""
    getEdgeGaussPoints(elem::Continuum, iedge::Int64)
    The element nodes are ordered as 
    #   4 ---- 3             #   4 --7-- 3
    #                        #   8   9   6 
    #   1 ---- 2             #   1 --5-- 2
    for porder=1     or          porder=2
    edge 1, 2, 3, 4 are (1,2), (2,3), (3,4), (4,1)
                    are (1,2,5), (2,3,6), (3,4,7), (4,1,8)

Returns the Gauss quadrature nodes of the element on its iedge-th edge in the undeformed domain
"""
function getEdgeGaussPoints(elem::Continuum, iedge::Int64)
    n = length(elem.elnodes)
    ngp = Int64(sqrt(length(elem.weights)))

    @assert(n == 4 || n == 9)

    n1, n2 = iedge, ((iedge+1)==5 ? 1 : iedge+1)
    loc_id = (n == 4 ? [n1, n2] : [n1, n2, iedge+4])

    x = elem.coords[loc_id, :]

    gnodes = zeros(ngp,2)

    _, hs = get1DElemShapeData(x, ngp)  

    gnodes = zeros(ngp,2)   
    for igp = 1:ngp
        gnodes[igp,:] = x' * hs[igp] 
    end

    return gnodes
end



function dofCount(elem::Continuum)
    return 2length(elem.elnodes)
end

"""
    commitHistory(elem::Continuum)

Updates the historic parameters in the material properties. 
"""
function commitHistory(elem::Continuum)
    for m in elem.mat 
        commitHistory(m)
    end
end