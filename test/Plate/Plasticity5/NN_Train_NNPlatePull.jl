using Optim
using LineSearches
stress_scale = 1.0e5
strain_scale = 1

include("nnutil.jl")

force_scale = 5.0
force_scales = [5.0]

testtype = "NeuralNetwork2D"
nntype = "piecewise"

# ! define H0
# Trained with nx, ny = 10, 5
H0 = [1.04167e6  2.08333e5  0.0      
      2.08333e5  1.04167e6  0.0      
      0.0        0.0        4.16667e5]/stress_scale


n_data = [100, 200, 201, 202, 203]

porder = 2
prop = Dict("name"=> testtype, "rho"=> 4.5, "nn"=>nn)


T = 0.1
NT = 200

# DNS computaional domain
fiber_size = 2
# nx_f, ny_f = 40*fiber_size, 20*fiber_size
nx_f, ny_f = 10*fiber_size, 5*fiber_size

# nx_f, ny_f = 12, 4

# homogenized computaional domain
# number of elements in each directions
nx, ny = 10, 5
# number of subelements in one element in each directions
sx_f, sy_f = div(nx_f,nx), div(ny_f,ny)

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


function compute_loss(tid, force_scale)
    nodes, EBC, g, gt, FBC, fext, ft, npoints, node_to_point = BoundaryCondition(tid, nx, ny, porder; force_scale=force_scale )
    domain = Domain(nodes, elements, ndofs, EBC, g, FBC, fext)
    state = zeros(domain.neqs)
    ∂u = zeros(domain.neqs)
    globdat = GlobalData(state,zeros(domain.neqs), zeros(domain.neqs),∂u, domain.neqs, gt, ft)
    assembleMassMatrix!(globdat, domain)
    # full_state_history, full_fext_history = read_data("$(@__DIR__)/Data/order$porder/$(tid)_$(force_scale)_$(fiber_size).dat")
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
    DynamicMatLawLoss(domain, globdat, state_history, fext_history, nn, Δt)
end


Δt = T/NT


nodes, _, _, _, _, _, _,_,_ = BoundaryCondition(n_data[1], nx, ny, porder)
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
        losses[k] = compute_loss(i, force_scale)/stress_scale
        k += 1
    end
end

@show stress_scale^2
loss = sum(losses)



















# function BFGS!(sess::PyObject, loss::PyObject, grads::Union{Array{T},Nothing,PyObject}, 
#     vars::Union{Array{PyObject},PyObject}; kwargs...) where T<:Union{Nothing, PyObject}
#     if isa(grads, PyObject); grads = [grads]; end
#     if isa(vars, PyObject); vars = [vars]; end
#     if length(grads)!=length(vars); error("ADCME: length of grads and vars do not match"); end
    
#     idx = ones(Bool, length(grads))
#     for i = 1:length(grads)
#         if isnothing(grads[i])
#             idx[i] = false
#         end
#     end
#     grads = grads[idx]
#     vars = vars[idx]
    
#     sizes = []
#     for v in vars
#         push!(sizes, size(v))
#     end
#     grds = vcat([tf.reshape(g, (-1,)) for g in grads]...)
#     vs = vcat([tf.reshape(v, (-1,)) for v in vars]...); x0 = run(sess, vs)
#     pl = placeholder(x0)
#     n = 0
#     assign_ops = PyObject[]
#     for (k,v) in enumerate(vars)
#         push!(assign_ops, assign(v, tf.reshape(pl[n+1:n+prod(sizes[k])], sizes[k])))
#         n += prod(sizes[k])
#     end
    
#     __loss = 0.0
#     __losses = Float64[]
#     function f(x)
#         run(sess, assign_ops, pl=>x)
#         __loss = run(sess, loss)
#         return __loss
#     end
    
#     function g!(G, x)
#         run(sess, assign_ops, pl=>x)
#         G[:] = run(sess, grds)
#     end
    
#     function callback(x)
#         push!(__losses, __loss)
#         false
#     end
    
#     Optim.optimize(f, g!, x0, Optim.LBFGS(alphaguess = InitialStatic(), linesearch=LineSearches.BackTracking(order=3)), 
#                    Optim.Options(show_trace=true, callback=callback, iterations=1000))
#     return __losses
# end



sess = tf.Session(); init(sess)
ADCME.load(sess, "$(@__DIR__)/Data/NNPreLSfit_$(idx)_ite5.mat")
#vars = get_collection()
for i = 1:50
    println("************************** Outer Iteration = $i ************************** ")
    BFGS!(sess, loss, 1000)
    #BFGS!(sess, loss, gradients(loss,vars), vars, iterations=1000)
    ADCME.save(sess, "Data/NN_Train_$(idx)_from_5_ite$(i).mat")
end


