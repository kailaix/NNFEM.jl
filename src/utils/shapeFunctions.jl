
using LinearAlgebra
export getShapeQuad4
function getShapeQuad4( ξ::Array{Float64,1} )
    #   gaussian point ordering:
    #   4 ---- 3
    #
    #   1 ---- 2
    #Check the dimension of physical space
    if length(ξ) != 2
        error("2D only")
    end


    sData       = zeros(4,3)

    #Calculate shape functions
    sData[1,1] = 0.25*(1.0-ξ[1])*(1.0-ξ[2])
    sData[2,1] = 0.25*(1.0+ξ[1])*(1.0-ξ[2])
    sData[3,1] = 0.25*(1.0+ξ[1])*(1.0+ξ[2])
    sData[4,1] = 0.25*(1.0-ξ[1])*(1.0+ξ[2])
    
    
    
    

    #Calculate derivatives of shape functions
    sData[1,2] = -0.25*(1.0-ξ[2])
    sData[2,2] =  0.25*(1.0-ξ[2])
    sData[3,2] =  0.25*(1.0+ξ[2])
    sData[4,2] = -0.25*(1.0+ξ[2])
    
    sData[1,3] = -0.25*(1.0-ξ[1])
    sData[2,3] = -0.25*(1.0+ξ[1])
    sData[3,3] =  0.25*(1.0+ξ[1])
    sData[4,3] =  0.25*(1.0-ξ[1])
    
    
    
    return sData
end



@doc """
    Return the Gauss quadrature points and weights in [-1,1]^2
""" -> 
function getIntegrationPoints(nPoints::Int64)
    if nPoints == 1
        q = [0.0]
        w = [2.0]
    elseif nPoints == 2
        q = sqrt(1. /3.)*[-1.;1.]
        w = [1.; 1]
    elseif nPoints == 3
        q = sqrt(3. /5.)*[ 0.; -1.; 1.]
        w = 1. /9. *[ 8.; 5.; 5.]
    elseif nPoints == 4
        q = [sqrt(3. / 7. - 2. /7. *sqrt(6. /5.)) ; -sqrt(3. / 7. - 2. /7. *sqrt(6. /5.)) ; sqrt(3. / 7. + 2. /7. *sqrt(6. /5.)) ; -sqrt(3. / 7. + 2. /7. *sqrt(6. /5.)) ]
        w = 1. / 36. * [18. + sqrt(30.); 18. + sqrt(30.); 18. - sqrt(30.); 18. - sqrt(30.)]
    end

    q2 = zeros(nPoints*nPoints, 2)
    w2 = zeros(nPoints*nPoints)
    for i = 1:nPoints
        for j = 1:nPoints
            n = (i-1)*nPoints + j
            q2[n, :] = [q[i]; q[j]]
            w2[n] = w[i]*w[j]
        end
    end
    # @show q2
    return q2, w2
end

@doc """
    :elemCoords 4x2 
""" ->
function getElemShapeData( elemCoords::Array{Float64} , nPoints::Int64 = 0 )

  @assert size(elemCoords)==(4,2)
  #elemType = getElemType( elemCoords )
    
  (intCrds,intWghts) = getIntegrationPoints( nPoints )
#   @show intCrds, intWghts
  dhdx = Array{Float64}[]
  weights = Float64[]
  hs = Array{Float64}[]
  for k = 1:length(intWghts)
    ξ = intCrds[k,:]
    weight = intWghts[k]
    # println(ξ)
    sData = getShapeQuad4(ξ)
    
    jac = elemCoords' * sData[:,2:end]
    push!(dhdx, sData[:,2:end] * inv( jac ))

    push!(weights, abs(det(jac)) * weight)
    push!(hs, sData[:,1])
  end
  

  return dhdx, weights, hs
end
