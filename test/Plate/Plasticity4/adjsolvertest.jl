
strain_scale = 1.0
stress_scale = 1.0e5;
tid = 200
porder = 2
T = 0.1
NT = 50

include("CommonFuncs.jl")
np = pyimport("numpy")

prop = Dict("name"=> "PlaneStress","rho"=> 4.5, "E"=> 1e+6, "nu"=> 0.2,
"sigmaY"=>0.97e+4, "K"=>1e+5)
H = PlaneStress(prop).H



nx, ny = 10,5

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
    ndofs = 2
    domain = Domain(nodes, elements, ndofs, EBC, g, FBC, fext)
    neqs = domain.neqs
    globdat = GlobalData(zeros(domain.neqs),zeros(domain.neqs), zeros(domain.neqs),zeros(domain.neqs), neqs, gt, ft)
    assembleMassMatrix!(globdat, domain)
    updateStates!(domain, globdat)

    J, state,strain, stress = ForwardNewmarkSolver(globdat, domain, theta, T, NT, strain_scale, stress_scale, obs_state)
end


function BackwardAdjoint(theta,  state, strain, stress, obs_state)
    ndofs = 2
    domain = Domain(nodes, elements, ndofs, EBC, g, FBC, fext)
    neqs = domain.neqs
    globdat = GlobalData(zeros(domain.neqs),zeros(domain.neqs), zeros(domain.neqs),zeros(domain.neqs), neqs, gt, ft)
    assembleMassMatrix!(globdat, domain)
    updateStates!(domain, globdat)
 
    dJ = BackwardNewmarkSolver(globdat, domain, theta, T, NT, state, strain, stress, strain_scale, stress_scale, obs_state)

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
obs_state = rand(Float64, NT+1, domain.neqs)
theta = [H[1,1], H[1,2], H[1,3], H[2,2], H[2,3], H[3,3]]/stress_scale


# J1, state1,_,_ = ForwardAdjoint(theta,  obs_state)
# state2 = ForwardSolve([H[1,1], H[1,2], H[1,3], H[2,2], H[2,3], H[3,3]],  obs_state)
# @show "Forward state error is ", norm(state2 - state1)
# @show "Forward J error is ", J1 - norm(state1[2:end,:] - obs_state[2:end,:])^2
# error()

function AdjointFunc(theta)
    J, state, strain, stress = ForwardAdjoint(theta,  obs_state)
    dJ = BackwardAdjoint(theta,  state, strain, stress, obs_state)
    return J, dJ'
end




theta = rand(704) *1.e-3
# ForwardAdjoint(theta,  obs_state)
# J, state = ForwardAdjoint(theta,  obs_state)
# dJ = BackwardAdjoint(theta,  state, obs_state)
gradtest(AdjointFunc, theta, scale=1.e-4)


