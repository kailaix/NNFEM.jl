mutable struct PlaneStrain
    H::Array{Float64}
    E::Float64
    ν::Float64
end

function PlaneStrain()
end

function getStress(d::PlaneStrain, deformation::Array{Float64})
end

function getTangent(d::PlaneStrain)
end

