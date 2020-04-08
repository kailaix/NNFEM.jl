# default values 
force_scale = 200.0 
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

testtype = "PlaneStress"

#ρ = 1153.4 kg/m^3 = 1153.4 kg/m^3 * 0.000076 m = 0.0876584 kg/m^2
#E = 944836855.0kg/m/s^2 * 0.000076m = 71807.60098kg/s^2 = 0.07180760098kg/ms^2
prop = Dict("name"=> testtype, "rho"=> 0.0876584, "E"=>0.07180760098, "nu"=>0.4)


NT = 1000
dt = 2.0e-3  #ms
T = NT * dt

nxc, nyc = 10, 5
nx, ny =  nxc*fiber_size, nyc*fiber_size
Lx, Ly = 0.2, 0.1 #m
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
        push!(elements,FiniteStrainContinuum(coords,elnodes, prop, 3))
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

@show norm(domain.fext)

Δt = T/NT


#Midpoint rule solver
ρ_oo = 1.0
αm = (2*ρ_oo - 1)/(ρ_oo + 1)
αf = ρ_oo/(ρ_oo + 1)

SolverInitial!(Δt, globdat, domain)
ω = EigenMode(Δt, globdat, domain)
@show "stable time step is ", 0.8 * 2/ω, " current time step is ", Δt
for i = 1:NT
    @info i, "/" , NT
    solver = NewmarkSolver(Δt, globdat, domain, αm, αf, 1e-4, 1e-6, 10)

end


# error()
if !isdir("$(@__DIR__)/Data")
    mkdir("$(@__DIR__)/Data")
end

if !isdir("$(@__DIR__)/Debug")
    mkdir("$(@__DIR__)/Debug")
end

if !isdir("$(@__DIR__)/Data/order$porder")
    mkdir("$(@__DIR__)/Data/order$porder")
end
if !isdir("$(@__DIR__)/Debug/order$porder")
    mkdir("$(@__DIR__)/Debug/order$porder")
end

write_data("$(@__DIR__)/Data/order$porder/$(tid)_$(force_scale)_$(fiber_size).dat", domain)
@save "Data/order$porder/domain$(tid)_$(force_scale)_$(fiber_size).jld2" domain

close("all")
frame = NT
# length scale(cm), stress scale(GPa), time scale(s)
scales = [100.0, 1.0/0.000076*1.0e-3, 1.0e-3]
visσ(domain, nx, ny,  domain.history["stress"][frame], domain.history["state"][frame]; scaling = scales)
xlabel("X (cm)")
ylabel("Y (cm)")
PyPlot.tight_layout()
savefig("Debug/order$porder/terminal$(tid)_$(force_scale)_$(fiber_size).png")


close("all")
Disp = zeros(Float64, NT+1, 5)
Disp[:,1] = LinRange(0, T, NT+1)
ux = [reshape(domain.history["state"][i][1:(nx*porder+1)*(ny*porder+1)], nx*porder+1, ny*porder+1)[end,end] for i = 1:length(domain.history["state"])]
uy = [reshape(domain.history["state"][i][(nx*porder+1)*(ny*porder+1)+1:end], nx*porder+1, ny*porder+1)[end,end] for i = 1:length(domain.history["state"])]
Disp[:,2], Disp[:,3] = ux, uy
ux = [reshape(domain.history["state"][i][1:(nx*porder+1)*(ny*porder+1)], nx*porder+1, ny*porder+1)[div((nx*porder+1),2),end] for i = 1:length(domain.history["state"])]
uy = [reshape(domain.history["state"][i][(nx*porder+1)*(ny*porder+1)+1:end], nx*porder+1, ny*porder+1)[div((nx*porder+1),2),end] for i = 1:length(domain.history["state"])]
Disp[:,4], Disp[:,5]  = ux, uy
plot(Disp[:, 1]*scales[3], Disp[:, 2]*scales[1], "--r",  label="Ref")
plot(Disp[:, 1]*scales[3], Disp[:, 3]*scales[1], "--y")
plot(Disp[:, 1]*scales[3], Disp[:, 4]*scales[1], "--b")
plot(Disp[:, 1]*scales[3], Disp[:, 5]*scales[1], "--g")

savefig("Debug/order$porder/disp$(tid)_$(force_scale)_$(fiber_size).png")