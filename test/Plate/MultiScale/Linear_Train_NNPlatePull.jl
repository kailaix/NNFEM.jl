stress_scale = 1.0e5
strain_scale = 1

include("nnutil.jl")
# reset_default_graph()
H0 = Variable(zeros(3,3))
H0 = H0 + H0'
# H0 = constant(H1/stress_scale)
testtype = "NeuralNetwork2D"
force_scales = [50.0]
nntype = "linear"
n_data = [100,200,201,202,203]
printstyled("training data: $n_data\n", color=:green)


# H0 = [4.5584e6   1.59544e6  0.0      
# 1.59544e6  4.5584e6   0.0      
# 0.0        0.0        1.48148e6]
# density 4.5*(1 - 0.25) + 3.2*0.25
fiber_fraction = 0.25
#todo
# fiber_fraction = 1.0
prop = Dict("name"=> testtype, "rho"=> 4.5*(1 - fiber_fraction) + 3.2*fiber_fraction, "nn"=>nn)


T = 0.05
NT = 100
fiber_size = 5
# DNS computaional domain
nx_f, ny_f = 80*fiber_size, 40*fiber_size
# nx_f, ny_f = 12, 4

# homogenized computaional domain
# number of elements in each directions
nx, ny = 10, 5
# number of subelements in one element in each directions
sx_f, sy_f = div(nx_f,nx), div(ny_f,ny)

porder = 2

ndofs = 2
fine_to_coarse = zeros(Int64, ndofs*(nx*porder+1)*(ny*porder+1))
for idof = 1:ndofs
for iy = 1:ny*porder+1
    for ix = 1:nx*porder+1
        fine_to_coarse[ix + (iy - 1)*(nx*porder+1) + (idof-1)*(nx*porder+1)*(ny*porder+1)] = 
        1 + (ix - 1) * sx_f + (iy - 1) * (nx_f*porder + 1) * sy_f + (nx_f*porder + 1)*(ny_f*porder + 1)*(idof - 1)
    end
end
end

# #todo only for first order
# function compute_fine_to_coarse_fext(tid)
#     @assert(porder == 1)
#     # Attention fix left
#     if div(tid,100)==1 # fix bottom
#         fine_to_coarse_fext = zeros(Int64, ndofs*(nx + 1)* ny)
#         for idof = 1:ndofs
#             for iy = 1:ny
#                 for ix = 1:nx + 1
#                     fine_to_coarse_fext[ix + (iy - 1)*(nx + 1) + (idof-1)*(nx + 1)*ny] =
#                      1 + (nx_f + 1) * (sy_f - 1)  + (iy - 1) * (nx_f + 1) * sy_f + (nx_f + 1)*ny_f*(idof - 1)
#                 end
#             end
#         end

#     elseif div(tid,100)==2 # fix left
#         fine_to_coarse_fext = zeros(Int64, ndofs*nx*(ny+1))
#         for idof = 1:ndofs
#             for iy = 1:ny+1
#                 for ix = 1:nx
#                     fine_to_coarse_fext[ix + (iy - 1)*(nx) + (idof-1)*(nx)*(ny+1)] = sx_f + (ix - 1) * sx_f + (iy - 1) * (nx_f) * sy_f + (nx_f)*(ny_f + 1)*(idof - 1)
#                 end
#             end
#         end
#     end
#     fine_to_coarse_fext
# end



# function compute_loss(tid)
#     local fscale
#     nodes, EBC, g, gt, FBC, fext, ft = BoundaryCondition(tid, nx, ny,porder)
#     domain = Domain(nodes, elements, ndofs, EBC, g, FBC, fext)
#     state = zeros(domain.neqs)
#     ∂u = zeros(domain.neqs)
#     globdat = GlobalData(state,zeros(domain.neqs), zeros(domain.neqs),∂u, domain.neqs, gt, ft)
#     assembleMassMatrix!(globdat, domain)
#     # @load "Data/domain$tid.jld2" domain
#     full_state_history, full_fext_history = read_data("$(@__DIR__)/Data/order$porder/$(tid)_$(force_scale)_$(fiber_size).dat")
#     #update state history and fext_history on the homogenized domain
#     state_history = [x[fine_to_coarse] for x in full_state_history]
#     #todo hard code the sy_f, it is on the right hand side
#     fine_to_coarse_fext = compute_fine_to_coarse_fext(tid)
    
#     if tid in [100, 300]
#         fscale = sx_f
#     elseif tid in [200, 201, 202, 203]
#         fscale = sy_f
#     end
#     fext_history = [x[fine_to_coarse_fext] * fscale for x in full_fext_history]
#     # @show size(hcat(domain.history["state"]...))
#     DynamicMatLawLoss(domain, globdat, state_history, fext_history, nn,Δt)
# end


function compute_loss(tid, force_scale)
    nodes, EBC, g, gt, FBC, fext, ft = BoundaryCondition(tid, nx, ny, porder, force_scale)
    domain = Domain(nodes, elements, ndofs, EBC, g, FBC, fext)
    state = zeros(domain.neqs)
    ∂u = zeros(domain.neqs)
    globdat = GlobalData(state,zeros(domain.neqs), zeros(domain.neqs),∂u, domain.neqs, gt, ft)
    assembleMassMatrix!(globdat, domain)
    # @load "Data/domain$tid.jld2" domain
    full_state_history, full_fext_history = read_data("$(@__DIR__)/Data/order$porder/$(tid)_$(force_scale)_$(fiber_size).dat")
    #update state history and fext_history on the homogenized domain
    state_history = [x[fine_to_coarse] for x in full_state_history]

    fext_history = []
    setNeumannBoundary!(domain, FBC, fext)
    for i = 1:NT
        globdat.time = Δt*i
        updateDomainStateBoundary!(domain, globdat)
        push!(fext_history, domain.fext[:])
    end
    DynamicMatLawLoss(domain, globdat, state_history, fext_history, nn,Δt)
end


Δt = T/NT


nodes, _, _, _, _, _, _ = BoundaryCondition(n_data[1], nx, ny, porder)
elements = []
for j = 1:ny
    for i = 1:nx 
        n = (nx*porder+1)*(j-1)*porder + (i-1)porder+1
        #element (i,j)
        if porder == 1
            #   4 ---- 3
            #
            #   1 ---- 2

            elnodes = [n, n + 1, n + 1 + (nx + 1), n + (nx + 1)]
        elseif porder == 2
            #   4 --7-- 3
            #   8   9   6 
            #   1 --5-- 2
            elnodes = [n, n + 2, n + 2 + 2*(2*nx+1),  n + 2*(2*nx+1), n+1, n + 2 + (2*nx+1), n + 1 + 2*(2*nx+1), n + (2*nx+1), n+1+(2*nx+1)]
        else
            error("polynomial order error, porder= ", porder)
        end
        coords = nodes[elnodes,:]
        push!(elements,SmallStrainContinuum(coords,elnodes, prop, 3))
    end
end

losses = Array{PyObject}(undef, length(n_data)*length(force_scales))
k = 1
for i in n_data
    global k
    for force_scale in force_scales
        losses[k] = compute_loss(i, force_scale)
        k += 1
    end
end
@show stress_scale^2
loss = sum(losses)/stress_scale^2
sess = Session(); init(sess)
# ADCME.load(sess, "$(@__DIR__)/Data/order1/learned_nn_5.0_1.mat")
# ADCME.load(sess, "Data/train_neural_network_from_fem.mat")
@info run(sess, loss)
# error()
for i = 1:100
    println("************************** Outer Iteration = $i ************************** ")
    BFGS!(sess, loss, 200)
    writedlm("$(@__DIR__)/Data/H0.mat", run(sess, H0))
end