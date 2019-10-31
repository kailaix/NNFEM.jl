using Optim
stress_scale = 1.0e5
strain_scale = 1.0
T = 0.1
NT = 200
include("CommonFuncs.jl")

force_scales = [5.0]

force_scale = 5.0

# ! define H0
# Trained with nx, ny = 10, 5
H0 = [1.04167e6  2.08333e5  0.0      
      2.08333e5  1.04167e6  0.0      
      0.0        0.0        4.16667e5]/stress_scale


n_data = [100, 200, 201, 202, 203]
n_data = [100]

@info "Thread number is ", Threads.nthreads() 
@info "Mulithreading requires: export JULIA_NUM_THREADS=", length(n_data)

# density 4.5*(1 - 0.25) + 3.2*0.25
#fiber_fraction = 0.25
#todo
porder = 2
prop = Dict("name"=> "PlaneStressPlasticity","rho"=> 4.5, "E"=> 1e+6, "nu"=> 0.2,
"sigmaY"=>0.97e+4, "K"=>1e+5)




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
        full_state_history, _ = read_data("$(@__DIR__)/Data/order$porder/$(tid)_$(force_scale)_$(fiber_size).dat")
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
    state::Array{Array{Float64}}
    strain::Array{Array{Float64}}
    stress::Array{Array{Float64}}

    function Buffer(n::Int64, ntheta::Int64, neqs::Array{Int64}, ngps::Int64, nstrain::Int64)
        J = zeros(n)
        dJ = [zeros(ntheta) for i = 1:n]
        state = [zeros(Float64, neqs[i]) for i = 1:n]
        strain = [zeros(Float64, ngps, nstrain) for i = 1:n]
        stress = [zeros(Float64, ngps, nstrain) for i = 1:n]
        new(J, dJ, state, strain, stress)
    end
end



function calculate_common!(theta, last_theta, buffer)
    if theta != last_theta
        @show theta
        copy!(last_theta, theta)

        for i = 1:length(n_data)
            #@show Threads.threadid()
            # todo: inplace 
            buffer.J[i], buffer.state[i], buffer.strain[i], buffer.stress[i] = ForwardNewmarkSolver(globdat_arr[i], domain_arr[i], theta, T, NT, strain_scale, stress_scale, obs_state_arr[i])
        end
    end
end

function f(theta, buffer, last_theta)   
    
    calculate_common!(theta, last_theta, buffer)
    return sum(buffer.J)
end

function g!(theta, storage, buffer, last_theta)
    calculate_common!(theta, last_theta, buffer)

    for i = 1:length(n_data)
        # todo: inplace 
        buffer.dJ[i] = BackwardNewmarkSolver(globdat_arr[i], domain_arr[i], theta, T, NT, buffer.state[i], buffer.strain[i], buffer.stress[i], strain_scale, stress_scale, obs_state_arr[i])
    end
    storage[:] = sum(buffer.dJ)
end



nstrain = 3
ngps_per_elem = length(domain_arr[1].elements[1].weights)
neles = domain_arr[1].neles
initial_theta = rand(704) * 1.e-3
neqs_arr = [domain_arr[i].neqs for i = 1:length(n_data)]
buffer = Buffer(length(n_data), length(initial_theta), neqs_arr, neles*ngps_per_elem, nstrain) # Preallocate an appropriate buffer
last_theta = similar(initial_theta)
# df = TwiceDifferentiable(x -> f(x, buffer, initial_theta),
#                                 (stor, x) -> g!(x, stor, buffer, last_theta))
# optimize(df, initial_theta, LBFGS())



# function AdjointFunc(theta)
#     dJ = similar(theta)
#     J = f(theta, buffer, last_theta)  
#     g!(theta, dJ, buffer, last_theta)
    
#     return J, dJ' 

# end
# gradtest(AdjointFunc, initial_theta, scale=1.e-4)


optimize(x -> f(x, buffer, initial_theta), 
        (stor, x) -> g!(x, stor, buffer, last_theta), 
        initial_theta, 
        LBFGS())



