include("nnutil.jl")
include("CommonFuncs.jl")
th = 1e-2 #dm
force_scale = 5.0/th #50
tid = 200
fiber_size = 1
porder = 2
stress_scale = 1.0e3
strain_scale = 1

if length(ARGS) == 4
   global tid = parse(Int64, ARGS[1])
   global force_scale = parse(Float64, ARGS[2])
   global fiber_size = parse(Int64, ARGS[3])
   global porder = parse(Int64, ARGS[4])
end
printstyled("force_scale=$force_scale, tid=$tid, fiber_size=$fiber_size, porder=$porder\n", color=:green)

if Sys.MACHINE=="x86_64-apple-darwin18.6.0"
    matplotlib.use("macosx")
end
np = pyimport("numpy")





Δt = 2.0e-4
NT = 10
T = Δt * NT #50.0 #500.0


# Step 1: make domain and globaldata
Lx, Ly = 1.0, 0.5
# number of elements in each directions
nxc, nyc = 10,5
nx, ny =  nxc*fiber_size, nyc*fiber_size
ngp = 3
prop = Dict("name"=> "PlaneStress","rho"=> 4.5, "E"=> 1e+4, "nu"=> 0.2)
Dc = PlaneStress(prop).H

nodes, EBC, g, gt, FBC, fext, ft, npoints, node_to_point = BoundaryCondition(tid, nx, ny, porder, Lx, Ly; force_scale=force_scale)
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
        push!(elements,SmallStrainContinuum(coords,elnodes, prop, ngp))
    end
end

ndofs = 2
domain = Domain(nodes, elements, ndofs, EBC, g, FBC, fext)
setGeometryPoints!(domain, npoints, node_to_point)

state = zeros(domain.neqs)
∂u = zeros(domain.neqs)
globdat = GlobalData(state,zeros(domain.neqs), zeros(domain.neqs),∂u, domain.neqs, gt, ft)

# Step 2: assemble global data mass matrix 
assembleMassMatrix!(globdat, domain)

# Step 3: compute_boundary_info to obtain time-depedent displacement and acceleration ubd, abd
ts = ExplicitSolverTime(Δt, NT)

ubd, abd = compute_boundary_info(domain, globdat, ts)


# Step 4: compute_external_force to obtain total external Fext 
Fext = compute_external_force(domain, globdat, ts)

# Step 5 get initial state, velo, acceleration with SovlerInitial
d0, v0, a0 = SolverInitial(Δt, globdat, domain)

# Step 6: Solve the equation
#d, v, a =  ExplicitSolver(globdat, domain, d0, v0, a0, Δt, NT, nn_law, Fext, ubd, abd; strain_type="small")

#TODO assume no prestress/prestrain
ngpt = getNGauss(domain)
σ0, ε0 = zeros(ngpt, 3), zeros(ngpt, 3)
d, v, a, σ, ε =  ExplicitSolver(globdat, domain, d0, v0, a0, σ0, ε0, Δt, NT, nn_nonassociated_plasticity, Fext, ubd, abd; strain_type="small")

# Step 7: Start optimization 
sess = Session(); init(sess)
dobs = matread("Data/order$(porder)/data$(tid).mat")["d"]
loss = sum((d - dobs)^2)
@show run(sess, loss)
BFGS!(sess, loss)
