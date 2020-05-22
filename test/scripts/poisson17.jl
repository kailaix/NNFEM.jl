using Distributed 
using ClusterManagers

σv = 0.001

for i = 1:5
    addprocs(SlurmManager(2))
end

@spawnat :any σv;

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
