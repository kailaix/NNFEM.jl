export getStiffAndForce, getInternalForce,
getBodyForce, getStrain, getMassMatrix,
getNodes, getGaussPoints, commitHistory

abstract type Continuum end 

@doc raw"""
    getBodyForce(elem::Continuum, fvalue::Array{Float64,2})

Returns the body force
$$\int_{e} \mathbf{f}\cdot \delta \mathbf{u} d \mathbf{x}$$
`fvalue` is a $n_{gauss}\times 2$ matrix. 
"""
function getBodyForce(elem::Continuum, fvalue::Array{Float64,2})
    n = dofCount(elem)
    fbody = zeros(Float64,n)
    out = Array{Float64}[]
    nnodes = length(elem.elnodes)
    for k = 1:length(elem.weights)
        fbody[1:nnodes] += elem.hs[k] * fvalue[k,1] * elem.weights[k]
        fbody[nnodes+1:2*nnodes] += elem.hs[k] * fvalue[k,2] * elem.weights[k]
    end
    return fbody
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

Returns the Gauss quadrature nodes of the element
"""
function getGaussPoints(elem::Continuum)
    x = elem.coords'
    gnodes = zeros(length(elem.weights),2)
    for k = 1:length(elem.weights)
        gnodes[k,:] = x * elem.hs[k] 
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