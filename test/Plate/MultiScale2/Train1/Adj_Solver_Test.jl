
strain_scale = 1.0
stress_scale = 1.0e5;
tid = 202
porder = 2
T = 0.1
NT = 200

include("../nnutil.jl")
np = pyimport("numpy")

prop = Dict("name"=> "PlaneStress","rho"=> 4.5, "E"=> 1e+6, "nu"=> 0.2,
"sigmaY"=>0.97e+4, "K"=>1e+5)
H = PlaneStress(prop).H

force_scale = 5.0

# DNS computaional domain
fiber_size = 5
# nx_f, ny_f = 40*fiber_size, 20*fiber_size
nx_f, ny_f = 80*fiber_size, 40*fiber_size

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

nodes, EBC, g, gt, FBC, fext, ft = BoundaryCondition(tid, nx, ny, porder)
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

############################################

function ForwardAdjoint(theta,  obs_state)
    J, dts, state_list,strain_list, stress_list = ForwardNewmarkSolver(globdat, domain, theta, T, NT, strain_scale, stress_scale, obs_state)
end


function BackwardAdjoint(theta, dts, state_list, strain_list, stress_list, obs_state)
 
    dJ = BackwardNewmarkSolver(globdat, domain, theta, T, NT, dts, state_list, strain_list, stress_list, strain_scale, stress_scale, obs_state)

end

################################3

function ForwardSolve(theta,  obs_state)
    ndofs = 2
    domain = Domain(nodes, elements, ndofs, EBC, g, FBC, fext)
    neqs = domain.neqs
    globdat = GlobalData(zeros(domain.neqs),zeros(domain.neqs), zeros(domain.neqs),zeros(domain.neqs), neqs, gt, ft)
    assembleMassMatrix!(globdat, domain)
    updateStates!(domain, globdat)

    adaptive_solver_args = Dict("Newmark_rho"=> 0.0, 
                          "Newton_maxiter"=>10, 
                          "Newton_Abs_Err"=>1e-4, 
                          "Newton_Rel_Err"=>1e-6, 
                          "damped_Newton_eta" => 1.0)
    globdat, domain, ts = AdaptiveSolver("NewmarkSolver", globdat, domain, T, NT, adaptive_solver_args)

    state2 = zeros(NT+1, neqs)
    for i=2:NT+1
        state2[i,:] = domain.history["state"][i][domain.dof_to_eq]
    end

    return state2
end


ndofs = 2
domain = Domain(nodes, elements, ndofs, EBC, g, FBC, fext)
neqs = domain.neqs
globdat = GlobalData(zeros(domain.neqs),zeros(domain.neqs), zeros(domain.neqs),zeros(domain.neqs), neqs, gt, ft)
assembleMassMatrix!(globdat, domain)

nstrain = 3
ngps_per_elem = length(domain.elements[1].weights)
neles = domain.neles

obs_state = zeros(Float64, NT+1, domain.neqs)
full_state_history, _ = read_data("../Data/order2/$(tid)_$(force_scale)_$(fiber_size).dat")
@assert length(full_state_history) == NT+1
for i = 1:NT+1
    obs_state[i,:] = (full_state_history[i][fine_to_coarse])[domain.dof_to_eq]
end


# obs_state = rand(Float64, NT+1, domain.neqs)
# theta = [H[1,1], H[1,2], H[1,3], H[2,2], H[2,3], H[3,3]]/stress_scale
# J1, dts, state_list, strain_list, stress_list  = ForwardAdjoint(theta,  obs_state)
# state = hcat(state_list...)
# state2 = ForwardSolve([H[1,1], H[1,2], H[1,3], H[2,2], H[2,3], H[3,3]],  obs_state)
# @show "Forward state error is ", norm(state2 - state')
# @show "Forward J error is ", J1 - norm(state'[2:end,:] - obs_state[2:end,:])^2
# dJ = BackwardNewmarkSolver(globdat, domain, theta, T, NT, dts, state_list, strain_list, stress_list, strain_scale, stress_scale, obs_state)
# error()

function AdjointFunc(theta)
    J, dts, state_list, strain_list, stress_list = ForwardAdjoint(theta,  obs_state)
    dJ = BackwardAdjoint(theta,  dts, state_list, strain_list, stress_list, obs_state)
    return J, dJ'
end



# theta = [H[1,1], H[1,2], H[1,3], H[2,2], H[2,3], H[3,3]]/stress_scale

theta = rand(1124) *1.e-3
config = [9, 20, 20, 20, 4]
#theta = convert_mat("nn2array", config,  "Data/piecewise/NNPreLSfit_$(idx)_spd_Chol_Orth_40.mat")


@load "Data/piecewise/theta_ite_idx0_from40_6.jld2" last_theta
theta = copy(last_theta)


#theta = rand(6) *1.e-3
# ForwardAdjoint(theta,  obs_state)
# J, state = ForwardAdjoint(theta,  obs_state)
# dJ = BackwardAdjoint(theta,  state, obs_state)
gradtest(AdjointFunc, theta, scale=1.e-4)


