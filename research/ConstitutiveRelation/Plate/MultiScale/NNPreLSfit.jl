stress_scale = 1.0e5
strain_scale = 1

include("nnutil.jl")

# H0 = constant(H1/stress_scale)
testtype = "NeuralNetwork2D"
force_scales = [5.0]
force_scale = 5.0

#nntype = "piecewise"
nntype = "piecewise"

# ! define H0
# Trained with nx, ny = 10, 5
# H0 = [1.26827e6       3.45169e5   -5187.35
#       3.45169e5       1.25272e6  -10791.7
#       -5187.35       -10791.7        536315.0]/stress_scale

H0 = [1335174.0968380707  326448.3267263398         0.0 
       326448.3267263398  1326879.2022994285        0.0 
                    0.0                 0.0     526955.763626241]/stress_scale

H0inv = inv(H0)

n_data = [100, 101, 102, 103, 104, 200, 201, 202, 203, 204]
porder = 2
# density 4.5*(1 - 0.25) + 3.2*0.25
fiber_fraction = 0.25
#todo
#fiber_fraction = 1.0
prop = Dict("name"=> testtype, "rho"=> 4.5*(1 - fiber_fraction) + 3.2*fiber_fraction, "nn"=>nn)


T = 200.0
NT = 200
Δt = T/NT

# DNS computaional domain
fiber_size = 5
# nx_f, ny_f = 40*fiber_size, 20*fiber_size
nx_f, ny_f = 80*fiber_size, 40*fiber_size

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


function approximate_stress(tid, force_scale, method)
    nodes, EBC, g, gt, FBC, fext, ft, npoints, node_to_point = BoundaryCondition(tid, nx, ny, porder; force_scale=force_scale )
    domain = Domain(nodes, elements, ndofs, EBC, g, FBC, fext)
    setGeometryPoints!(domain, npoints, node_to_point)

    state = zeros(domain.neqs)
    ∂u = zeros(domain.neqs)
    globdat = GlobalData(state,zeros(domain.neqs), zeros(domain.neqs),∂u, domain.neqs, gt, ft)
    assembleMassMatrix!(globdat, domain)

    # full_state_history, full_fext_history = read_data("$(@__DIR__)/Data/order$porder/$(tid)_$(force_scale)_$(fiber_size).dat")
    full_state_history, full_fext_history = read_data("Data/order$porder/$(tid)_$(force_scale)_$(fiber_size).dat")
    

    #update state history and fext_history on the homogenized domain
    state_history = [x[fine_to_coarse] for x in full_state_history]
    fext_history = []
    setConstantNodalForces!(domain, FBC, fext)
    for i = 1:NT
        globdat.time = Δt*i
        updateTimeDependentEssentialBoundaryCondition!(domain, globdat)
        fext = getExternalForce(domain, globaldat)
        push!(fext_history, fext)
    end


    E_all, S_all = LSfittingStress(domain, globdat, hcat(state_history...), hcat(fext_history...), Δt, method)

end

nodes, _ = BoundaryCondition(n_data[1], nx, ny, porder)
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



#################### Gradient test
# tid = n_data[1]
# method = "Constant"
# nodes, EBC, g, gt, FBC, fext, ft, npoints, node_to_point = BoundaryCondition(tid, nx, ny, porder; force_scale=force_scale )
# domain = Domain(nodes, elements, ndofs, EBC, g, FBC, fext)
# setGeometryPoints!(domain, npoints, node_to_point)
# function f(S_comp)
    
#     stress = reshape(S_comp, 3, div(size(S_comp,1),3))'|>Array
    
#     LSfittingStressHelper(domain, stress, method)
# end
# nS_comp = (method == "Constant") ? domain.neles : domain.npoints;
# S_comp = rand(Float64, 3*nS_comp)
# gradtest(f, S_comp)
#########################


#################### S_all compare with real S_all
# method = "Linear"
# tid = n_data[1]
# E_all, S_all = approximate_stress(tid, force_scale, method)
# @load "Data/order$porder/domain$(tid)_$(force_scale)_1.jld2" domain
# S_all_real = domain.history["stress"]
# E_all_real = domain.history["strain"]
# for it = 1:NT
#     @show "E_all[", it, "] error is ", norm(E_all[it+1,:,:] - E_all_real[it]), " / ", norm(E_all_real[it])
#     @show "S_all[", it, "] error is ", norm(S_all[it+1,:,:] - S_all_real[it]), " / ", norm(S_all_real[it])
# end
#########################



loss = constant(0.0)

method = "Linear"
#for force_scale in force_scales
for tid in n_data
    E_all, S_all = approximate_stress(tid, force_scale, method)
    ngp = size(E_all,2)


    X = zeros(Float64, (NT-1)*ngp, 9)
    Y = zeros(Float64, (NT-1)*ngp, 3)
    k = 1
    for it = 2:NT
        for igp = 1:ngp
            X[k,:] = [E_all[it,igp,:]; E_all[it-1,igp,:]; S_all[it-1,igp,:]]#ε, ε0, σ0
            Y[k,:] = S_all[it,igp,:]
            k = k + 1
        end
    end

    x = constant(X)
    y = nn(x[:,1:3], x[:,4:6], x[:,7:9])
    
    global loss
    loss += mean((y - Y)^2) #/stress_scale^2
end
#end

sess = Session(); init(sess)
@show run(sess, loss)
# ADCME.load(sess, "Data/NNPreLSfit.mat")
for i = 1:50
    BFGS!(sess, loss, 1000)
    @show "Data/$(nntype)/NNPreLSfit_$(idx)_$(H_function)_$(i).mat"
    ADCME.save(sess, "Data/$(nntype)/NNPreLSfit_$(idx)_$(H_function)_$(i).mat")
end