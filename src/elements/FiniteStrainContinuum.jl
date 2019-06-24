export FiniteStrainContinuum
mutable struct FiniteStrainContinuum
    mat  # constitutive law
    elnodes::Array{Int64}   # the node indices in this finite element
    props::Dict{String, Any}
    coords::Array{Float64}
    dhdx::Array{Array{Float64}}  # 4nPointsx2 matrix
    weights::Array{Float64} 
    hs::Array{Array{Float64}}
end

function FiniteStrainContinuum(coords::Array{Float64}, elnodes::Array{Int64}, props::Dict{String, Any})
    name = props["name"]
    if name=="PlaneStrain"
        mat = PlaneStrain(props)
    else
        error("Not implemented yet: $name")
    end
    dhdx, weights, hs = getElemShapeData( coords, 2 )
    FiniteStrainContinuum(mat, elnodes, props, coords, dhdx, weights, hs)
end

function getStiffAndForce(self::FiniteStrainContinuum, state::Array{Float64}, Dstate::Array{Float64})
    ndofs = dofCount(self); 
    nnodes = length(self.elnodes)
    fint = zeros(Float64, ndofs)
    stiff = zeros(Float64, ndofs,ndofs)
    out = Array{Float64}[]
    u = state[1:nnodes]; v = state[nnodes+1:2*nnodes]
    for k = 1:length(self.weights)
        g1 = self.dhdx[k][:,1]; g2 = self.dhdx[k][:,2]
        
        ux = u'*g1; uy = u'*g2; vx = v'*g1; vy = v'*g2
        # compute  ∂E∂u.T, 8 by 3 array 
        ∂E∂u = [g1+ux*g1 uy*g2    g2 + g2*ux+g1*uy;
                vx*g1    g2+vy*g2 g1 + g1*vy+g2*vx;] 
        
        E = [ux+0.5*(ux*ux+vx*vx); vy+0.5*(uy*uy+vy*vy); uy+vx+ux*uy+vx*vy]

        S, dS_dE = getStress(self.mat, E)

        fint += ∂E∂u * S * self.weights[k] # 1x8
        
        S∂∂E∂∂u = [g1*g1'*S[1]+g2*g2'*S[2]+g1*g2'*S[3]+g2*g1'*S[3] zeros(nnodes, nnodes);
                  zeros(nnodes, nnodes)  g1*g1'*S[1]+g2*g2'*S[2]+g1*g2'*S[3]+g2*g1'*S[3]]
        
        stiff += (∂E∂u * dS_dE * ∂E∂u' + S∂∂E∂∂u)*self.weights[k] # 8x8
    end
    return fint, stiff
end

function getInternalForce(self::FiniteStrainContinuum, state::Array{Float64}, Dstate::Array{Float64})
    n = dofCount(self)
    fint = zeros(Float64,n)
    out = Array{Float64}[]
    u = state[1:4]; v = state[5:8]
    for k = 1:length(self.weights)
        g1 = self.dhdx[k][:,1]; g2 = self.dhdx[k][:,2]
        ux = u'*g1; uy = u'*g2; vx = v'*g1; vy = v'*g2 
        @info "dudx", ux, uy, vx, vy
        # compute  ∂E∂u.T, 8 by 3 array
        ∂E∂u = [g1+ux*g1 uy*g2    g2 + g2*ux+g1*uy;
                vx*g1    g2+vy*g2 g1 + g1*vy+g2*vx;] 

        E = [ux+0.5*(ux*ux+vx*vx); vy+0.5*(uy*uy+vy*vy); uy+vx+ux*uy+vx*vy]
        S,_ = getStress(self.mat,E)
        @info "ε", E, "σ", S
        fint += ∂E∂u * S * self.weights[k] # 1x8
        @info "fint_l", ∂E∂u * S * self.weights[k], self.weights[k]
    end
    return fint
end

function getMassMatrix(self::FiniteStrainContinuum)
    ndofs = dofCount(self)
    nnodes = length(self.elnodes)
    rho = self.mat.ρ
    mass = zeros(ndofs,ndofs)
    for k = 1:length(self.weights)
        mass += [self.hs[k]*self.hs[k]' zeros(nnodes, nnodes)
                 zeros(nnodes, nnodes)  self.hs[k]*self.hs[k]']  * rho * self.weights[k]
    end
    lumped = sum(mass, dims=2)
    mass, lumped
end


function getNodes(self::FiniteStrainContinuum)
    return self.elnodes
end

function dofCount(self::FiniteStrainContinuum)
    return 2length(self.elnodes)
end


