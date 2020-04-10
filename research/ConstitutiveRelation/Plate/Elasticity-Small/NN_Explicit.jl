# default values 
force_scale = 200.0 
tid = 400
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



stress_scale = 1.0
strain_scale = 1.0
nntype = "ae_scaled"
idx = 1
H_function = "None"
include("nnutil.jl")

testtype = "NeuralNetwork2D"


s = ae_to_code("Data/NNLearn_$(nntype)_ite10.mat", nntype)
eval(Meta.parse(s))

prop = Dict("name"=> testtype, "rho"=> 0.0876584, "nn"=>post_nn)

NT = Int64(1e5)
dt = 2.0e-3  #ms
T = NT * dt

nx, ny = 10, 5
Lx, Ly = 0.2, 0.1 #m
nodes, EBC, g, gt, FBC, fext, ft, npoints, node_to_point = BoundaryCondition(tid, nx, ny,porder, Lx, Ly; force_scale=force_scale)

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
        push!(elements,SmallStrainContinuum(coords,elnodes, prop, 3))
    end
end



domain = Domain(nodes, elements, ndofs, EBC, g, FBC, fext)
state = zeros(domain.neqs)
∂u = zeros(domain.neqs)
globdat = GlobalData(state,zeros(domain.neqs), zeros(domain.neqs),∂u, domain.neqs, gt, ft)

assembleMassMatrix!(globdat, domain)
updateStates!(domain, globdat)



Δt = T/NT

#explicit solver
SolverInitial!(Δt, globdat, domain)

ω = EigenMode(Δt, globdat, domain)
@show "stable time step is ", 0.8 * 2/ω, " current time step is ", Δt
for i = 1:NT
    @info i, "/" , NT
    solver = ExplicitSolver(Δt, globdat, domain)
    if i%10 == 0
        ω = EigenMode(Δt, globdat, domain)
        @show "stable time step is ", 0.8 * 2/ω, " current time step is ", Δt
    end

end

# error()
if !isdir("$(@__DIR__)/Data")
    mkdir("$(@__DIR__)/Data")
end

if !isdir("$(@__DIR__)/Debug")
    mkdir("$(@__DIR__)/Debug")
end

if !isdir("$(@__DIR__)/Data/$nntype")
    mkdir("$(@__DIR__)/Data/$nntype")
end
if !isdir("$(@__DIR__)/Debug/$nntype")
    mkdir("$(@__DIR__)/Debug/$nntype")
end

write_data("$(@__DIR__)/Data/$nntype/$(tid)_$(force_scale)_$(fiber_size).dat", domain)
@save "Data/$nntype/domain$(tid)_$(force_scale)_$(fiber_size).jld2" domain

close("all")
frame = NT
# length scale(cm), stress scale(GPa), time scale(s)
scales = [100.0, 1.0/0.000076*1.0e-3, 1.0e-3]
visσ(domain, nx, ny,  domain.history["stress"][frame], domain.history["state"][frame]; scaling = scales)
xlabel("X (cm)")
ylabel("Y (cm)")
PyPlot.tight_layout()
savefig("Debug/$nntype/terminal$(tid)_$(force_scale)_$(fiber_size).png")


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

savefig("Debug/$nntype/disp$(tid)_$(force_scale)_$(fiber_size).png")