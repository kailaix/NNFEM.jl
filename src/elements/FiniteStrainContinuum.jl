mutable struct FiniteStrainContinuum
    mat
    elnodes::Array{Int64}
    props::Dict{String, Any}
end

function getTangentStiffness(self::FiniteStrainContinuum, coords::Array{Float64}, state::Array{Float64}, Dstate::Array{Float64})
    n = self.dofCount()

    sData = getElemShapeData( coords )
    
end

function getInternalForce(d::FiniteStrainContinuum, coords::Array{Float64}, state::Array{Float64}, Dstate::Array{Float64})
end

function getMassMatrix(d::FiniteStrainContinuum, coords::Array{Float64})
end

function getNodes(d::FiniteStrainContinuum, iele::Int64)
end

function getNodes(d::FiniteStrainContinuum)
end


