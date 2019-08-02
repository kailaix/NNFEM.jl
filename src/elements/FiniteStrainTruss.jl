export FiniteStrainTruss
mutable struct FiniteStrainTruss
    eledim::Int64
    mat  # constitutive law
    elnodes::Array{Int64}   # the node indices in this finite element
    props::Dict{String, Any}
    coords::Array{Float64}
    weights::Array{Float64}
    hs::Array{Array{Float64}} 
    strain::Array{Float64}
    l0::Float64                  # length in the undeformed shape
    A0::Float64                  # cross section area of the undeformed shape
    rot_mat::Array{Float64}      # 4 by 4 rotation matrix
end

function FiniteStrainTruss(coords::Array{Float64}, elnodes::Array{Int64}, props::Dict{String, Any}, ngp::Int64=2)
    eledim = 1
    weights, hs = get1DElemShapeData( coords, ngp )
    nGauss = length(weights)
    name = props["name"]
    if name=="Elasticity1D"
        mat = [Elasticity1D(props) for i = 1:nGauss]
    elseif name=="Plasticity1D"
        mat = [Plasticity1D(props) for i = 1:nGauss]
    elseif name=="Viscoplasticity1D"
        mat = [Viscoplasticity1D(props) for i = 1:nGauss]
    elseif name=="NeuralNetwork1D"
        mat = [NeuralNetwork1D(props) for i = 1:nGauss]
    elseif name=="PathDependent1D"
        mat = [PathDependent1D(props) for i = 1:nGauss]
    else
        error("Not implemented yet: $name")
    end
    strain = Array{Float64}(undef, length(weights))

    dx, dy = coords[2,1] - coords[1,1], coords[2,2] - coords[1,2]
    l0 = sqrt(dx^2 + dy^2)
    cosα, sinα = dx/l0, dy/l0 
    rot_mat = [cosα   0    sinα   0;
               0     cosα    0    sinα;
               -sinα  0    cosα   0;
               0     -sinα   0    cosα]

    FiniteStrainTruss(eledim, mat, elnodes, props, coords, weights, hs, strain, l0, props["A0"], rot_mat)
end


function getStiffAndForce(self::FiniteStrainTruss, state::Array{Float64}, Dstate::Array{Float64}, Δt::Float64)
    ndofs = dofCount(self); 
    nnodes = length(self.elnodes)
    fint = zeros(Float64, ndofs)
    stiff = zeros(Float64, ndofs,ndofs)
    
    
    rot_mat = self.rot_mat
    #rotate displacement to the local coordinate
    lu = rot_mat * state
    Dlu = rot_mat * Dstate

    A0, l0 = self.A0, self.l0

    E = (lu[2]-lu[1])/l0 + 0.5*((lu[2]-lu[1])/l0)^2 + 0.5*((lu[4]-lu[3])/l0)^2
    DE = (Dlu[2]-Dlu[1])/l0 + 0.5*((Dlu[2]-Dlu[1])/l0)^2 + 0.5*((Dlu[4]-Dlu[3])/l0)^2
    # compute  ∂E∂u.T, 4 by 1 array 
    ∂E∂u =  [-1/l0+(lu[1]-lu[2])/l0;  1/l0+(lu[2]-lu[1])/l0; (lu[3]-lu[4])/l0; (lu[4]-lu[3])/l0]
    ∂∂E∂∂u =  [1/l0  -1/l0     0        0;
              -1/l0   1/l0     0        0;   
               0       0      1/l0    -1/l0;       
               0       0     -1/l0    1/l0]   

    for k = 1:length(self.weights)

        # #@show E, DE
        S, dS_dE = getStress(self.mat[k], E, DE, Δt)
        
        self.strain[k] = S

        fint += A0 * self.weights[k] * S * ∂E∂u  # 4x1

        @info "state ", state , "lu", lu, "S ", S, "wdEdu ", A0 * self.weights[k] * ∂E∂u
        
        stiff += A0 * self.weights[k] * (dS_dE * ∂E∂u * ∂E∂u' + S * ∂∂E∂∂u) # 4x4
    end

    #rotate back to global coordinate

    fint = rot_mat' * fint
    stiff = rot_mat' * stiff * rot_mat

    return fint, stiff
end

function getInternalForce(self::FiniteStrainTruss, state::Array{Float64}, Dstate::Array{Float64}, Δt::Float64)
end

function getMassMatrix(self::FiniteStrainTruss)
    ndofs = dofCount(self)
    nnodes = length(self.elnodes)
    mass = zeros(ndofs,ndofs)

    A0 = self.A0
    for k = 1:length(self.weights)
        rho = self.mat[k].ρ

        mass += [self.hs[k]*self.hs[k]' zeros(nnodes, nnodes)
                 zeros(nnodes, nnodes)  self.hs[k]*self.hs[k]']  * A0 * rho * self.weights[k]
    end
    lumped = sum(mass, dims=2)
    #@show mass
    mass, lumped
end

function getStrain(self::FiniteStrainTruss, state::Array{Float64})
    ndofs = dofCount(self); 
    nnodes = length(self.elnodes)
    fint = zeros(Float64, ndofs)
    stiff = zeros(Float64, ndofs,ndofs)
    
    
    rot_mat = self.rot_mat
    #rotate displacement to the local coordinate
    lu = rot_mat * state

    A0, l0 = self.A0, self.l0

    nGauss = length(self.weights)
    E = zeros(nGauss, 1)
    w∂E∂u = zeros(nGauss, 4, 1)


    for k = 1:nGauss
        # compute  ∂E∂u.T, 8 by 3 array 
        E[k,1] = (lu[2]-lu[1])/l0 + 0.5*((lu[2]-lu[1])/l0)^2 + 0.5*((lu[4]-lu[3])/l0)^2
        w∂E∂u[k,:,:] = [-1/l0+(lu[1]-lu[2])/l0;  1/l0+(lu[2]-lu[1])/l0; (lu[3]-lu[4])/l0; (lu[4]-lu[3])/l0] * A0 * self.weights[k]
    end
    @info "state ", state , "lu", lu , "w∂E∂u[k,:,:] ", w∂E∂u[1,:,:]

    return E, w∂E∂u
end


function getNodes(self::FiniteStrainTruss)
    return self.elnodes
end

function dofCount(self::FiniteStrainTruss)
    return 2length(self.elnodes)
end

function commitHistory(self::FiniteStrainTruss)
    for m in self.mat 
        commitHistory(m)
    end
end