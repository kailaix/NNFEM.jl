th = 1e-2 #dm
force_scale = 4.0/th #50
tid = 200
fiber_size = 1
porder = 2
if length(ARGS) == 4
   global tid = parse(Int64, ARGS[1])
   global force_scale = parse(Float64, ARGS[2])
   global fiber_size = parse(Int64, ARGS[3])
   global porder = parse(Int64, ARGS[4])
end
printstyled("force_scale=$force_scale, tid=$tid, fiber_size=$fiber_size, porder=$porder\n", color=:green)
include("CommonFuncs.jl")
if Sys.MACHINE=="x86_64-apple-darwin18.6.0"
    matplotlib.use("macosx")
end
np = pyimport("numpy")

prop = Dict("name"=> "PlaneStressPlasticity","rho"=> 4.5, "E"=> 1e+4, "nu"=> 0.2,
"sigmaY"=>0.97e+2, "K"=>1e+3)


Δt = 2.0e-4
NT = 1000
T = Δt * NT #50.0 #500.0


nxc, nyc = 10,5
nx, ny =  nxc*fiber_size, nyc*fiber_size
Lx, Ly = 1.0, 0.5
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
        push!(elements,SmallStrainContinuum(coords,elnodes, prop, 3))
    end
end


ndofs = 2
domain = Domain(nodes, elements, ndofs, EBC, g, FBC, fext)
setGeometryPoints!(domain, npoints, node_to_point)

state = zeros(domain.neqs)
∂u = zeros(domain.neqs)
globdat = GlobalData(state,zeros(domain.neqs), zeros(domain.neqs),∂u, domain.neqs, gt, ft)

assembleMassMatrix!(globdat, domain)
updateStates!(domain, globdat)


ρ_oo = 0.0
αm = (2*ρ_oo - 1)/(ρ_oo + 1)
αf = ρ_oo/(ρ_oo + 1)

#explicit solver
SolverInitial!(Δt, globdat, domain)

ω = EigenMode(Δt, globdat, domain)
@show "stable time step is ", 0.8 * 2/ω, " current time step is ", Δt
for i = 1:NT
    @info i, "/" , NT
    #solver = ExplicitSolver(Δt, globdat, domain)
    solver = NewmarkSolver(Δt, globdat, domain, αm, αf, 1e-4, 1e-6, 10)
    if i%10 == 0
        ω = EigenMode(Δt, globdat, domain)
        @show "stable time step is ", 0.8 * 2/ω, " current time step is ", Δt
    end
end


# todo write data
if !isdir("$(@__DIR__)/Data/order$porder")
    mkdir("$(@__DIR__)/Data/order$porder")
end
if !isdir("$(@__DIR__)/Debug/order$porder")
    mkdir("$(@__DIR__)/Debug/order$porder")
end
# 2NT × 2domain.nnodes
d_obs, a_obs = Array(hcat(domain.history["state"]...)'), Array(hcat(domain.history["acc"]...)')
matwrite("Data/order$porder/data$(tid).mat", Dict("d"=>d_obs, "a"=>a_obs))

close("all")
visσ(domain)
axis("equal")
savefig("Debug/order$porder/terminal$(tid)_$(force_scale)_$(fiber_size).png")

close("all")
ux = [reshape(domain.history["state"][i][1:(nx*porder+1)*(ny*porder+1)], ny*porder+1, nx*porder+1)[end,end] for i = 1:length(domain.history["state"])]
plot(ux)
savefig("Debug/order$porder/ux$(tid)_$(force_scale)_$(fiber_size).png")

close("all")
uy = [reshape(domain.history["state"][i][(nx*porder+1)*(ny*porder+1)+1:end], ny*porder+1, nx*porder+1)[end,end] for i = 1:length(domain.history["state"])]
plot(uy)
savefig("Debug/order$porder/uy$(tid)_$(force_scale)_$(fiber_size).png")

