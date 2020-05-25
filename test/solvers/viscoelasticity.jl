using Revise
using NNFEM
using PyPlot


ViscoelasticitySolver(globdat::GlobalData, domain::Domain,
    d0::Union{Array{Float64, 1}, PyObject}, 
    v0::Union{Array{Float64, 1}, PyObject}, 
    a0::Union{Array{Float64, 1}, PyObject}, 
    σ0::Union{Array{Float64, 2}, PyObject}, 
    ε0::Union{Array{Float64, 2}, PyObject}, 
    Δt::Float64, NT::Int64, 
    μ::Union{Array{Float64, 1}, PyObject}, 
    λ::Union{Array{Float64, 1}, PyObject}, 
    η::Union{Array{Float64, 1}, PyObject}, 
    Fext::Union{Array{Float64, 2}, PyObject, Missing}=missing,
    ubd::Union{Array{Float64, 2}, PyObject, Missing}=missing,
    abd::Union{Array{Float64, 2}, PyObject, Missing}=missing; ρ::Float64 = 0.0)