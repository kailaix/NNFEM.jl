mutable struct Domain
    nnodes::Int64
    nodes::Array{Float64}
    neles::Int64
    elem::Array
    ndofs::Int64
    ndim::Int64
    state::Array{Float64}
    Dstate::Array{Float64}
end

function Domain(nodes::Array{Float64}, elements::Array, ndofs::Int64, ndims::Int64, EBC::Array{Int64}, g::Float64)
end

function setBoundary(d::Domain, EBC::Array{Int64}, g::Array{Float64})

end

function updateStates(d::Domain, state::Array{Float64}, Dstate::Array{Float64}, time::Float64)
end

function getCoords(d::Domain, el_nodes::Array{Int64})
end

function getEqns(d::Domain, iele::Array{Int64})
end

function getState(d::Domain, el_dofs::Array{Int64})
end


mutable struct GlobalData
    state::Array{Float64}
    Dstate::Array{Float64}
    velo::Array{Float64}
    acce::Array{Float64}
    finit::Array{Float64}
    fext::Array{Float64}
    time::Float64
end

function GlobalData()
end