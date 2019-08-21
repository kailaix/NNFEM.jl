using Revise
using Test 
using NNFEM
using PyCall
using PyPlot
using JLD2
using ADCME
using LinearAlgebra
# reset_default_graph()
include("nnutil.jl")
testtype = "NeuralNetwork2D"
nntype = "piecewise"

# density 4.5*(1 - 0.25) + 3.2*0.25
prop = Dict("name"=> testtype, "rho"=> 4.5*(1 - 0.25) + 3.2*0.25, "nn"=>nn)

# DNS computaional domain
#nx_f, ny_f = 600, 100
nx_f, ny_f = 120, 20
# homogenized computaional domain
# number of elements in each directions
nx, ny =  12, 5
# number of subelements in one element in each directions
sx_f, sy_f = nx_f/nx, ny_f/ny

fine_to_coarse = zeros(Int64, ndofs*(nx+1)*(ny+1))
for idof = 1:ndofs
for iy = 1:ny+1
    for ix = 1:nx+1
        fine_to_coarse[ix + (iy - 1)*(nx+1) + (idof-1)*(nx+1)*(ny+1)] = 1 + (ix - 1) * sx_f + (iy - 1) * (nx_f + 1) * sy_f + (nx_f + 1)*(ny_f + 1)*(idof - 1)
    end
end
end

# Attention fix left 
fine_to_coarse_fext = zeros(Int64, ndofs*nx*(ny+1))
for idof = 1:ndofs
for iy = 1:ny+1
    for ix = 1:nx
        fine_to_coarse_fext[ix + (iy - 1)*(nx) + (idof-1)*(nx)*(ny+1)] = sx_f + (ix - 1) * sx_f + (iy - 1) * (nx_f) * sy_f + (nx_f)*(ny_f + 1)*(idof - 1)
    end
end
end

# DOF = zeros(Int64, nx_f+1, ny_f+1)
# k = 1
# for i = 1:nx_f+1
#     for j = 1:ny_f+1    
#         DOF[i, j] = k
#         global k += 1
#     end
# end


nnodes, neles = (nx + 1)*(ny + 1), nx*ny
Lx, Ly = 3.0, 0.5
x = np.linspace(0.0, Lx, nx + 1)
y = np.linspace(0.0, Ly, ny + 1)

X, Y = np.meshgrid(x, y)
nodes = zeros(nnodes,2)
nodes[:,1], nodes[:,2] = X'[:], Y'[:]
ndofs = 2

# set boundary conditions
EBC, g = zeros(Int64, nnodes, ndofs), zeros(nnodes, ndofs)
EBC[collect(1:nx+1:(nx+1)*(ny+1)), :] .= -1 # fix left

function ggt(t)
    return zeros(sum(EBC.==-2)), zeros(sum(EBC.==-2))
end
gt = ggt

#pull or compress on the right
FBC, fext = zeros(Int64, nnodes, ndofs), zeros(nnodes, ndofs)

function training_fext(nx, ny, θ; alpha = 1.0)
    F = 5e6 * alpha   #elastic 3e6 ; plasticity starts from 4e6
    
    FBC[collect(nx+1:nx+1:(nx+1)*(ny+1)), 1] .= -1
    FBC[collect(nx+1:nx+1:(nx+1)*(ny+1)), 2] .= -1
    
    fext[collect(nx+1:nx+1:(nx+1)*(ny+1)), 1] .= F * cos(θ)
    fext[collect(nx+1:nx+1:(nx+1)*(ny+1)), 2] .= F * sin(θ)
    fext[nx+1, 1] /= 2
    fext[nx+1, 2] /= 2
    fext[(nx+1)*(ny+1), 1] /= 2
    fext[(nx+1)*(ny+1), 2] /= 2
    FBC, fext
end

FBC, fext = training_fext(nx, ny, 0, alpha =sy_f)

#force load function
function fft(t)
    f = 1.0e6 
end
ft = fft

elements = []
for j = 1:ny
    for i = 1:nx 
        n = (nx+1)*(j-1) + i
        elnodes = [n, n + 1, n + 1 + (nx + 1), n + (nx + 1)]
        coords = nodes[elnodes,:]
        push!(elements,SmallStrainContinuum(coords,elnodes, prop, 3))
    end
end



# Time 
T = 0.001
NT = 100
Δt = T/NT



stress_scale = 1.0e5
strain_scale = 1
domain = Domain(nodes, elements, ndofs, EBC, g, FBC, fext)


state = zeros(domain.neqs)
∂u = zeros(domain.neqs)
globdat = GlobalData(state,zeros(domain.neqs), zeros(domain.neqs),∂u, domain.neqs, gt, ft)
assembleMassMatrix!(globdat, domain)

losses = Array{PyObject}(undef, n_data)
for i = 1:n_data
    full_state_history, full_fext_history = read_data("$(@__DIR__)/Data/$i.dat")
    #update state history and fext_history on the homogenized domain
    state_history = [x[fine_to_coarse] for x in full_state_history]
    #todo hard code the sy_f, it is on the right hand side
    
    fext_history = [x[fine_to_coarse_fext] * sy_f for x in full_fext_history]

    #fext_history = [fext[:] for k = 1:length(full_fext_history)]
    
    # @show  fext_history[1], fext[:]
    # @assert fext_history[1] == fext[:]

    losses[i] = DynamicMatLawLoss(domain, globdat, state_history, fext_history, nn,Δt)
end
loss = sum(losses)/stress_scale^2

sess = Session(); init(sess)
# ADCME.load(sess, "$(@__DIR__)/Data/learned_nn.mat")
# ADCME.load(sess, "Data/train_neural_network_from_fem.mat")
# @show run(sess, loss)
# error()
BFGS!(sess, loss, 1500)
# ADCME.save(sess, "$(@__DIR__)/Data/train_neural_network_from_fem.mat")
# ADCME.load(sess, "$(@__DIR__)/Data/train_neural_network_from_fem.mat")
# BFGS!(sess, loss, 5000)
# ADCME.save(sess, "$(@__DIR__)/Data/train_neural_network_from_fem.mat")
# BFGS!(sess, loss, 5000)
ADCME.save(sess, "$(@__DIR__)/Data/train_neural_network_from_fem.mat")

# * test neural network
close("all")
@load "$(@__DIR__)/Data/domain1.jld2" domain
X, Y = prepare_strain_stress_data2D(domain)
x = constant(X)
y = nn(X[:,1:3], X[:,4:6], X[:,7:9])
init(sess)
ADCME.load(sess, "$(@__DIR__)/Data/train_neural_network_from_fem.mat")

try
    global O = run(sess, y)
catch
    global O = y 
end
using Random; Random.seed!(233)
VisualizeStress2D(Y, O, 20)
savefig("$(@__DIR__)/Debug/trained_nn.png")
