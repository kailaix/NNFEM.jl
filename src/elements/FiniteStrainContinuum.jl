mutable struct FiniteStrainContinuum
    mat
    elnodes::Array{Int64}
    props::Dict{String, Any}
end

function getTangentStiffness(d::FiniteStrainContinuum, coords::Array{Float64}, state::Array{Float64}, Dstate::Array{Float64})
end

function getInternalForce(d::FiniteStrainContinuum, coords::Array{Float64}, state::Array{Float64}, Dstate::Array{Float64})
end

function getMassMatrix(d::FiniteStrainContinuum, coords::Array{Float64})
end



