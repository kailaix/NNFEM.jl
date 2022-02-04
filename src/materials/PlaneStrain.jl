export  PlaneStrain

@doc raw"""
    PlaneStrain

Creates a plane strain element

- `H`: Linear elasticity matrix, $3\times3$
- `E`: Young's modulus
- `ν`: Poisson's ratio 
- `ρ`: density 
- `σ0`: stress at the **last** time step 
- `σ0_`: (for internal use), stress to be updated in `commitHistory`
- `ε0`: strain at the **last** time step 
- `ε0_`: (for internal use), strain to be updated in `commitHistory`

# Example
```julia
prop = Dict("name"=> "PlaneStrain", "rho"=> 0.0876584, "E"=>0.07180760098, "nu"=>0.4)
mat = PlaneStrain(prop)
```
"""
mutable struct PlaneStrain
    H::Array{Float64}
    E::Float64
    ν::Float64
    ρ::Float64
    σ0::Array{Float64} 
    σ0_::Array{Float64} 
    ε0::Array{Float64} 
    ε0_::Array{Float64} 
end

"""
    PlaneStrain(prop::Dict{String, Any})

`prop` should contain at least the following three fields: `E`, `nu`, `rho`
"""
function PlaneStrain(prop::Dict{String, Any})
    E = prop["E"]; ν = prop["nu"]; ρ = prop["rho"]
    H = zeros(3,3)
    H[1,1] = E*(1. -ν)/((1+ν)*(1. -2. *ν));
    H[1,2] = H[1,1]*ν/(1-ν);
    H[2,1] = H[1,2];
    H[2,2] = H[1,1];
    H[3,3] = H[1,1]*0.5*(1. -2. *ν)/(1. -ν);
    σ0 = zeros(3); σ0_ = zeros(3); ε0 = zeros(3); ε0_ = zeros(3)
    PlaneStrain(H, E, ν, ρ, σ0, σ0_,ε0,ε0_)
end

function getStress(self::PlaneStrain, strain::Array{Float64}, Dstrain::Array{Float64}, Δt::Float64 = 0.0)
    sigma = self.H * strain
    self.σ0_ = copy(sigma)
    self.ε0_ = copy(strain)
    return sigma, self.H
end

function getTangent(self::PlaneStrain)
    self.H
end

function commitHistory(self::PlaneStrain)
    self.σ0 = self.σ0_
    self.ε0 = self.ε0_
end
