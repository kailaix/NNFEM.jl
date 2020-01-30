stress_scale = 1.0e5
strain_scale = 1.0

tid= 200

include("nnutil.jl")

# H0 = constant(H1/stress_scale)
testtype = "NeuralNetwork2D"


force_scales = [5.0]

force_scale = 5.0

H0 = [1335174.0968380707  326448.3267263398 0.0 
       326448.3267263398 1326879.2022994285 0.0 
            0.0                0.0     526955.763626241]/stress_scale
      
H0inv = inv(H0)


n_data = [200]
porder = 2
# density 4.5*(1 - 0.25) + 3.2*0.25
fiber_fraction = 0.25
#todo
#fiber_fraction = 1.0
prop = Dict("name"=> testtype, "rho"=> 4.5*(1 - fiber_fraction) + 3.2*fiber_fraction, "nn"=>nn)

T = 0.1
NT = 200


# homogenized computaional domain
# number of elements in each directions
nx, ny = 10, 5

porder = 2

nodes, EBC, g, gt, FBC, fext, ft = BoundaryCondition(tid, nx, ny,porder; force_scale=force_scale)

ndofs=2
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
        push!(elements,SmallStrainContinuum(coords, elnodes, prop, 3))
    end
end





        
domain = Domain(nodes, elements, ndofs, EBC, g, FBC, fext)
state = zeros(domain.neqs)
∂u = zeros(domain.neqs)
globdat = GlobalData(state,zeros(domain.neqs), zeros(domain.neqs),∂u, domain.neqs, gt, ft)

assembleMassMatrix!(globdat, domain)
updateStates!(domain, globdat)

nstrain = 3
ngps_per_elem = length(domain.elements[1].weights)
neles = domain.neles

obs_state = zeros(Float64, NT+1, domain.neqs)
state = zeros(Float64, NT+1, domain.neqs)
strain = zeros(Float64,NT+1, neles * ngps_per_elem, nstrain) 
stress = zeros(Float64,NT+1, neles * ngps_per_elem, nstrain) 


config = [9, 20, 20, 20, 20, 4]
@load  "Data/theta_ite10_bfgs.jld2" last_theta

ForwardNewmarkSolver(globdat, domain, last_theta, T, NT, strain_scale, stress_scale, 
                     obs_state, state, strain, stress, -1.0, 0.0,  1e-4, 
                     1e-6, 10) 
   

ts = LinRange(0, T, NT+1)

state_all = zeros(Float64, NT+1, 2*(nx*porder + 1)*(ny*porder + 1))
for i = 1:NT+1
    state_all[i, :] = domain.state
    state_all[i, domain.eq_to_dof] = state[i,:]
end

close("all")
ux = [reshape(state_all[i,1:(nx*porder+1)*(ny*porder+1)], ny*porder+1, nx*porder+1)[1,end] for i = 1:NT+1]
plot(ts, ux)
savefig("Debug/order$porder/test_ux_$(tid)_$force_scale.png")

close("all")
uy = [reshape(state_all[i,(nx*porder+1)*(ny*porder+1)+1:end], ny*porder+1, nx*porder+1)[1,end] for i = 1:NT+1]
plot(ts, uy)
savefig("Debug/order$porder/test_uy_$(tid)_$force_scale.png")
