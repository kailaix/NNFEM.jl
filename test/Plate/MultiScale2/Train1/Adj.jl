using Optim, LineSearches
stress_scale = 1.0e5
strain_scale = 1.0
T = 0.1
NT = 200

include("../nnutil.jl")

# H0 = constant(H1/stress_scale)
testtype = "NeuralNetwork2D"


force_scales = [5.0]

force_scale = 5.0

H0 = [1335174.0968380707  326448.3267263398 0.0 
       326448.3267263398 1326879.2022994285 0.0 
            0.0                0.0     526955.763626241]/stress_scale
      
H0inv = inv(H0)


n_data = [100,  200, 201, 202, 203]
porder = 2
# density 4.5*(1 - 0.25) + 3.2*0.25
fiber_fraction = 0.25
#todo
#fiber_fraction = 1.0
prop = Dict("name"=> testtype, "rho"=> 4.5*(1 - fiber_fraction) + 3.2*fiber_fraction, "nn"=>nn)


T = 0.1
NT = 200

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



#####################Step 1 preprocess data
function PreprocessData(n_data, NT)
    globdat_arr = []
    domain_arr = []
    obs_state_arr = []
    
    for (id, tid) in enumerate(n_data)

        
         

        nodes, EBC, g, gt, FBC, fext, ft = BoundaryCondition(tid, nx, ny, porder)
        domain = Domain(nodes, elements, ndofs, EBC, g, FBC, fext)
        neqs = domain.neqs
        globdat = GlobalData(zeros(domain.neqs),zeros(domain.neqs), zeros(domain.neqs),zeros(domain.neqs), neqs, gt, ft)
        assembleMassMatrix!(globdat, domain)


        obs_state = zeros(Float64, NT+1, domain.neqs)
        full_state_history, _ = read_data("../Data/order$porder/$(tid)_$(force_scale)_$(fiber_size).dat")
        @assert length(full_state_history) == NT+1
        for i = 1:NT+1
            obs_state[i,:] = (full_state_history[i][fine_to_coarse])[domain.dof_to_eq]
        end


        push!(domain_arr, domain)
        push!(globdat_arr, globdat)
        push!(obs_state_arr, obs_state)
    end

    return globdat_arr, domain_arr, obs_state_arr
end


globdat_arr, domain_arr, obs_state_arr = PreprocessData(n_data, NT)


mutable struct Buffer
    J::Array{Float64}
    dJ::Array{Array{Float64}}
    dts::Array{Any,1}
    state::Array{Any,1}
    strain::Array{Any,1}
    stress::Array{Any,1}

    function Buffer(n::Int64, ntheta::Int64, NT::Int64, neqs::Array{Int64}, ngps::Int64, nstrain::Int64)
        J = zeros(n)
        dJ = [zeros(ntheta) for i = 1:n]
        dts = [[] for i = 1:n]
        state = [[] for i = 1:n]
        strain = [[] for i = 1:n]
        stress = [[] for i = 1:n]
        new(J, dJ, dts, state, strain, stress)
    end
end

function calculate_common!(theta, last_theta, buffer)
    @show " theta norm ", norm(theta), " last_theta norm ", norm(last_theta)
    if theta != last_theta
        
        copy!(last_theta, theta)

        Threads.@threads for i = 1:length(n_data)
        #for i = 1:length(n_data)
            #@show Threads.threadid()
            # todo: inplace 
            @show n_data[i]
            buffer.J[i], buffer.dts[i], buffer.state[i], buffer.strain[i], buffer.stress[i] = 
            ForwardNewmarkSolver(globdat_arr[i], domain_arr[i], theta, T, NT, strain_scale, stress_scale, obs_state_arr[i])

            #if buffer.J[i]  == Inf
            #    break
            #end
        end
    end
end

function f(theta, buffer, last_theta)   
    calculate_common!(theta, last_theta, buffer)

    @show buffer.J
    J = sum(buffer.J)
    
    @show "function evaluation: |J|=", norm(J)
    
    return J
end

function g!(theta, storage, buffer, last_theta)
    calculate_common!(theta, last_theta, buffer)
    
    
    Threads.@threads for i = 1:length(n_data)
        # todo: inplace 
        buffer.dJ[i] = BackwardNewmarkSolver(globdat_arr[i], domain_arr[i], theta, T, NT, buffer.dts[i], buffer.state[i], buffer.strain[i], buffer.stress[i], strain_scale, stress_scale, obs_state_arr[i])
    end
    
    storage[:] = sum(buffer.dJ)

    @show "gradient evaluation: |dJ/dtheta|=", norm( storage )

    storage[:] ./= max(1.0, norm(storage))
end



nstrain = 3
ngps_per_elem = length(domain_arr[1].elements[1].weights)
neles = domain_arr[1].neles


config = [9, 20, 20, 20, 4]
initial_theta = convert_mat("nn2array", config,  "Data/piecewise/NNPreLSfit_$(idx)_spd_Chol_Orth_40.mat")

#initial_theta = 1.0e-3*rand((9+1)*20 + (20+1)*20 + (20+1)*20 + (20+1)*4)

#@load "Data/theta_ite_0_5.jld2" last_theta
#initial_theta = copy(last_theta)

neqs_arr = [domain_arr[i].neqs for i = 1:length(n_data)]
buffer = Buffer(length(n_data), length(initial_theta), NT, neqs_arr, neles*ngps_per_elem, nstrain) # Preallocate an appropriate buffer



@load "Data/piecewise/theta_ite_idx0_from40_6.jld2" last_theta
theta = copy(last_theta)

last_theta .= 0.0

J =  f(theta, buffer, last_theta)  
@show "J1 = ", J

storage = similar(theta)
dJ = g!(theta, storage, buffer, last_theta)


J =  f(theta - 0.001*dJ, buffer, last_theta) 
@show "J2 = ", J


J =  f(theta + 0.001*dJ, buffer, last_theta) 
@show "J3 = ", J