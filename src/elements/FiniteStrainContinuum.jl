include("Continuum.jl")
export FiniteStrainContinuum


"""
    FiniteStrainContinuum

Constructs a finite strain element. 

- `eledim`: spatial dimension of the element (default = 2).
- `mat`: constitutive law, a length `#elem` vector of materials such as [`PlaneStress`](@ref)
- `elnodes`: the node indices in this finite element, an integer array 
- `props`: property dictionary 
- `coords`: coordinates of the vertices of the element
- `dhdx`, `weights`, `hs`: data for integral 
- `stress`: stress at each quadrature points

# Example
```julia
#   Local degrees of freedom 
#   4 ---- 3
#
#   1 ---- 2

nx = 10
ny = 5
h = 0.1
element = FiniteStrainContinuum[]
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
        push!(element, FiniteStrainContinuum(coords,elnodes, prop, ngp))
    end
end
```
"""
mutable struct FiniteStrainContinuum <: Continuum
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

Base.show(io::IO, z::FiniteStrainContinuum) = print(io, "FiniteStrainContinuum with $(length(z.weights)) Gauss quadrature points and $(z.props["name"]) material.")


"""
    FiniteStrainContinuum(coords::Array{Float64}, elnodes::Array{Int64}, props::Dict{String, Any}, ngp::Int64=2)
"""
function FiniteStrainContinuum(coords::Array{Float64}, elnodes::Array{Int64}, props::Dict{String, Any}, ngp::Int64=2)
    eledim = 2
    dhdx, weights, hs = get2DElemShapeData( coords, ngp )
    nGauss = length(weights)
    name = props["name"]
    if name=="PlaneStrain"
        mat = [PlaneStrain(props) for i = 1:nGauss]
    elseif name=="Scalar1D"
        mat = [Scalar1D(props) for i = 1:nGauss]
    elseif name=="PlaneStress"
        mat = [PlaneStress(props) for i = 1:nGauss]
    elseif name=="PlaneStressPlasticity"
        mat = [PlaneStressPlasticity(props) for i = 1:nGauss]
    elseif name=="PlaneStrainViscoelasticityProny"
        mat = [PlaneStrainViscoelasticityProny(props) for i = 1:nGauss]
    elseif name=="PlaneStressViscoelasticityProny"
        mat = [PlaneStressViscoelasticityProny(props) for i = 1:nGauss]
    elseif name=="PlaneStressIncompressibleRivlinSaunders"
        mat = [PlaneStressIncompressibleRivlinSaunders(props) for i = 1:nGauss]
    elseif name=="NeuralNetwork2D"
        mat = [NeuralNetwork2D(props) for i = 1:nGauss]
    else
        error("Not implemented yet: $name")
    end
    strain = Array{Array{Float64}}(undef, length(weights))
    FiniteStrainContinuum(eledim, mat, elnodes, props, coords, dhdx, weights, hs, strain)
end

function getStiffAndForce(self::FiniteStrainContinuum, state::Array{Float64}, Dstate::Array{Float64}, Δt::Float64)
    ndofs = dofCount(self); 
    nnodes = length(self.elnodes)
    fint = zeros(Float64, ndofs)
    stiff = zeros(Float64, ndofs,ndofs)
    out = Array{Float64}[]
    u = state[1:nnodes]; v = state[nnodes+1:2*nnodes]
    Du = Dstate[1:nnodes]; Dv = Dstate[nnodes+1:2*nnodes]
    for k = 1:length(self.weights)
        g1 = self.dhdx[k][:,1]; g2 = self.dhdx[k][:,2]
        
        ux = u'*g1; uy = u'*g2; vx = v'*g1; vy = v'*g2
        Dux = Du'*g1; Duy = Du'*g2; Dvx = Dv'*g1; Dvy = Dv'*g2
        # compute  ∂E∂u.T, 8 by 3 array 
        ∂E∂u = [g1+ux*g1 uy*g2    g2 + g2*ux+g1*uy;
                vx*g1    g2+vy*g2 g1 + g1*vy+g2*vx;] 
        
        E = [ux+0.5*(ux*ux+vx*vx); vy+0.5*(uy*uy+vy*vy); uy+vx+ux*uy+vx*vy]
        DE = [Dux+0.5*(Dux*Dux+Dvx*Dvx); Dvy+0.5*(Duy*Duy+Dvy*Dvy); Duy+Dvx+Dux*Duy+Dvx*Dvy]

        # #@show "+++",E, DE
        S, dS_dE = getStress(self.mat[k], E, DE, Δt)
        # error()
        self.stress[k] = S

        fint += ∂E∂u * S * self.weights[k] # 1x8
        
        S∂∂E∂∂u = [g1*g1'*S[1]+g2*g2'*S[2]+g1*g2'*S[3]+g2*g1'*S[3] zeros(nnodes, nnodes);
                  zeros(nnodes, nnodes)  g1*g1'*S[1]+g2*g2'*S[2]+g1*g2'*S[3]+g2*g1'*S[3]]
        
        stiff += (∂E∂u * dS_dE * ∂E∂u' + S∂∂E∂∂u)*self.weights[k] # 8x8
    end
    return fint, stiff
end

function getInternalForce(self::FiniteStrainContinuum, state::Array{Float64}, Dstate::Array{Float64}, Δt::Float64)
    n = dofCount(self)
    fint = zeros(Float64,n)
    out = Array{Float64}[]
    nnodes = length(self.elnodes)
    u = state[1:nnodes]; v = state[nnodes+1:2*nnodes]
    Du = Dstate[1:nnodes]; Dv = Dstate[nnodes+1:2*nnodes]
    for k = 1:length(self.weights)
        g1 = self.dhdx[k][:,1]; g2 = self.dhdx[k][:,2]
        ux = u'*g1; uy = u'*g2; vx = v'*g1; vy = v'*g2 
        Dux = Du'*g1; Duy = Du'*g2; Dvx = Dv'*g1; Dvy = Dv'*g2
        # compute  ∂E∂u.T, 8 by 3 array
        ∂E∂u = [g1+ux*g1 uy*g2    g2 + g2*ux+g1*uy;
                vx*g1    g2+vy*g2 g1 + g1*vy+g2*vx;] 

        E = [ux+0.5*(ux*ux+vx*vx); vy+0.5*(uy*uy+vy*vy); uy+vx+ux*uy+vx*vy]
        DE = [Dux+0.5*(Dux*Dux+Dvx*Dvx); Dvy+0.5*(Duy*Duy+Dvy*Dvy); Duy+Dvx+Dux*Duy+Dvx*Dvy]
        S, _ = getStress(self.mat[k],E, DE, Δt)

        self.stress[k] = S
        
        fint += ∂E∂u * S * self.weights[k] # 1x8
    end
    return fint
end



function getStrain(self::FiniteStrainContinuum, state::Array{Float64})
    ndofs = dofCount(self); 
    nnodes = length(self.elnodes)
    fint = zeros(Float64, ndofs)
    stiff = zeros(Float64, ndofs,ndofs)
    out = Array{Float64}[]
    u = state[1:nnodes]; v = state[nnodes+1:2*nnodes]

    nGauss = length(self.weights)
    E = zeros(nGauss, 3)
    w∂E∂u = zeros(nGauss, 2nnodes, 3)
    for k = 1:length(self.weights)
        g1 = self.dhdx[k][:,1]; g2 = self.dhdx[k][:,2]
        
        ux = u'*g1; uy = u'*g2; vx = v'*g1; vy = v'*g2
        # compute  ∂E∂u.T, 8 by 3 array 
        ∂E∂u = [g1+ux*g1 uy*g2    g2 + g2*ux+g1*uy;
                vx*g1    g2+vy*g2 g1 + g1*vy+g2*vx;]  # 8x3
        E[k,:] = [ux+0.5*(ux*ux+vx*vx); vy+0.5*(uy*uy+vy*vy); uy+vx+ux*uy+vx*vy] # 3x1
        w∂E∂u[k,:,:] = ∂E∂u*self.weights[k] # 8x3
    end

    return E, w∂E∂u
end




function  getForceAndDforceDstress(self::FiniteStrainContinuum, state::Array{Float64},  
    stress::Array{Float64,2})
    ndofs = dofCount(self); 
    nnodes = length(self.elnodes)
    nStrain = 3
    nGauss = length(self.weights)
    fint = zeros(Float64, ndofs)
    dfint_dstress = zeros(Float64,  ndofs, nGauss * nStrain)

    u = state[1:nnodes]; v = state[nnodes+1:2*nnodes]

    for k = 1:length(self.weights)
        g1 = self.dhdx[k][:,1]; g2 = self.dhdx[k][:,2]

        ux = u'*g1; uy = u'*g2; vx = v'*g1; vy = v'*g2
        # compute  ∂E∂u, 3 by 2nnodes array 
        ∂E∂u = [g1+ux*g1 uy*g2    g2 + g2*ux+g1*uy;
                vx*g1    g2+vy*g2 g1 + g1*vy+g2*vx;]

        
        S = stress[k, :]

        self.stress[k] = S

        fint += ∂E∂u * S * self.weights[k] # 1x8
        
        dfint_dstress[:, (k-1)*nStrain+1:k*nStrain] = ∂E∂u * self.weights[k]

    end
    return fint , dfint_dstress
end

