#=
Study the effect of σv 
=#
using Distributed 
using ClusterManagers

σv = 0.001
ndata = 20

if length(ARGS)==2
    σv = parse(Float64, ARGS[1])
    ndata = parse(Int64, ARGS[2])
end

for i = 1:5
    addprocs(SlurmManager(2))
end

[@spawnat i σv for i in workers()]
[@spawnat i ndata for i in workers()]

@info "Generating data..."
include("poisson1.jl")
@info "Training Neural Network..."
include("poisson13.jl")
@info "Compute Hessian"
include("poisson14.jl")
@info "Computing Posterior"
include("poisson15.jl")
@info "Prediction"
include("poisson16.jl")
