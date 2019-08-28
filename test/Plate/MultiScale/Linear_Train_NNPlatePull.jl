include("nnutil.jl")
# reset_default_graph()
H0 = Variable(zeros(3,3))
H0 = H0 + H0'
# H0 = constant(H1/stress_scale)
testtype = "NeuralNetwork2D"
force_scale = 50
nntype = "linear"
n_data = [100]


# density 4.5*(1 - 0.25) + 3.2*0.25
fiber_fraction = 0.25
#todo
#fiber_fraction = 1.0
prop = Dict("name"=> testtype, "rho"=> 4.5*(1 - fiber_fraction) + 3.2*fiber_fraction, "nn"=>nn)


T = 0.05
NT = 100

# DNS computaional domain
nx_f, ny_f = 40*10, 20*10
# nx_f, ny_f = 12, 4

# homogenized computaional domain
# number of elements in each directions
nx, ny = 20, 20
# number of subelements in one element in each directions
sx_f, sy_f = div(nx_f,nx), div(ny_f,ny)

ndofs = 2
fine_to_coarse = zeros(Int64, ndofs*(nx+1)*(ny+1))
for idof = 1:ndofs
for iy = 1:ny+1
    for ix = 1:nx+1
        fine_to_coarse[ix + (iy - 1)*(nx+1) + (idof-1)*(nx+1)*(ny+1)] = 
        1 + (ix - 1) * sx_f + (iy - 1) * (nx_f + 1) * sy_f + (nx_f + 1)*(ny_f + 1)*(idof - 1)
    end
end
end


function compute_fine_to_coarse_fext(tid)
    # Attention fix left

    if div(tid,100)==1 # fix bottom
        fine_to_coarse_fext = zeros(Int64, ndofs*(nx + 1)* ny)
        for idof = 1:ndofs
            for iy = 1:ny
                for ix = 1:nx + 1
                    fine_to_coarse_fext[ix + (iy - 1)*(nx + 1) + (idof-1)*(nx + 1)*ny] =
                     1 + (nx_f + 1) * (sy_f - 1)  + (iy - 1) * (nx_f + 1) * sy_f + (nx_f + 1)*ny_f*(idof - 1)
                end
            end
        end

    elseif div(tid,100)==2 # fix left
        fine_to_coarse_fext = zeros(Int64, ndofs*nx*(ny+1))
        for idof = 1:ndofs
            for iy = 1:ny+1
                for ix = 1:nx
                    fine_to_coarse_fext[ix + (iy - 1)*(nx) + (idof-1)*(nx)*(ny+1)] = sx_f + (ix - 1) * sx_f + (iy - 1) * (nx_f) * sy_f + (nx_f)*(ny_f + 1)*(idof - 1)
                end
            end
        end
    end
    fine_to_coarse_fext
end



function compute_loss(tid)
    local fscale
    nodes, EBC, g, gt, FBC, fext, ft = BoundaryCondition(tid, nx, ny)
    domain = Domain(nodes, elements, ndofs, EBC, g, FBC, fext)
    state = zeros(domain.neqs)
    ∂u = zeros(domain.neqs)
    globdat = GlobalData(state,zeros(domain.neqs), zeros(domain.neqs),∂u, domain.neqs, gt, ft)
    assembleMassMatrix!(globdat, domain)
    # @load "Data/domain$tid.jld2" domain
    full_state_history, full_fext_history = read_data("$(@__DIR__)/Data/LinearData/$(tid)_$force_scale.0.dat")
    #update state history and fext_history on the homogenized domain
    state_history = [x[fine_to_coarse] for x in full_state_history]
    #todo hard code the sy_f, it is on the right hand side
    fine_to_coarse_fext = compute_fine_to_coarse_fext(tid)
    
    if tid in [100, 300]
        fscale = sx_f
    elseif tid in [200, 201, 202, 203]
        fscale = sy_f
    end
    fext_history = [x[fine_to_coarse_fext] * fscale for x in full_fext_history]
    # @show size(hcat(domain.history["state"]...))
    DynamicMatLawLoss(domain, globdat, state_history, fext_history, nn,Δt)
end




Δt = T/NT
stress_scale = 1.0
strain_scale = 1

nodes, _, _, _, _, _, _ = BoundaryCondition(n_data[1], nx, ny)
elements = []
for j = 1:ny
    for i = 1:nx 
        n = (nx+1)*(j-1) + i
        elnodes = [n, n + 1, n + 1 + (nx + 1), n + (nx + 1)]
        coords = nodes[elnodes,:]
        push!(elements,SmallStrainContinuum(coords,elnodes, prop, 3))
    end
end

losses = Array{PyObject}(undef, length(n_data))
for (k, i) in enumerate(n_data)
    losses[k] = compute_loss(i)
end
loss = sum(losses)/stress_scale^2

sess = Session(); init(sess)
# ADCME.load(sess, "$(@__DIR__)/Data/learned_nn.mat")
# ADCME.load(sess, "Data/train_neural_network_from_fem.mat")
@info run(sess, loss)
# error()
BFGS!(sess, loss, 2000)
# ADCME.save(sess, "$(@__DIR__)/Data/train_neural_network_from_fem.mat")
# ADCME.load(sess, "$(@__DIR__)/Data/train_neural_network_from_fem.mat")
# BFGS!(sess, loss, 5000)
# ADCME.save(sess, "$(@__DIR__)/Data/train_neural_network_from_fem.mat")
# BFGS!(sess, loss, 5000)
ADCME.save(sess, "$(@__DIR__)/Data/train_neural_network_from_fem.mat")
error()
# * test neural network
close("all")
@load "$(@__DIR__)/Data/domain100.jld2" domain
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
